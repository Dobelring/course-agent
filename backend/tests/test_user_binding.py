"""权限绑定：UserBoundTool 强制覆盖 user_id，从机制上防越权（Agent 安全核心）。"""
import asyncio
import json
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agent.engine import _needs_user_id, bind_user, pretty_tool_name


class _EchoArgs(BaseModel):
    user_id: int
    query: str = ""


class EchoTool(BaseTool):
    """回显收到的参数，用于断言 user_id 是否被强制覆盖。"""

    name: str = "echo"
    description: str = "回显参数"
    args_schema: Type[BaseModel] = _EchoArgs

    def _run(self, user_id: int, query: str = "") -> str:
        return json.dumps({"user_id": user_id, "query": query}, ensure_ascii=False)


class _NoUserArgs(BaseModel):
    query: str = ""


class PlainTool(BaseTool):
    """不接收 user_id 的工具，不应被包装。"""

    name: str = "plain"
    description: str = "无 user_id 参数"
    args_schema: Type[BaseModel] = _NoUserArgs

    def _run(self, query: str = "") -> str:
        return query


def test_needs_user_id_detects_pydantic_schema():
    assert _needs_user_id(EchoTool()) is True
    assert _needs_user_id(PlainTool()) is False


def test_needs_user_id_handles_dict_schema():
    class DictSchemaTool:
        args_schema = {"properties": {"user_id": {"type": "integer"}}}

    class NoUserDictTool:
        args_schema = {"properties": {"q": {"type": "string"}}}

    assert _needs_user_id(DictSchemaTool()) is True
    assert _needs_user_id(NoUserDictTool()) is False


def test_needs_user_id_handles_none_schema():
    class NoneSchemaTool:
        args_schema = None

    assert _needs_user_id(NoneSchemaTool()) is False


def test_bind_user_wraps_only_tools_needing_user_id():
    tools = bind_user([EchoTool(), PlainTool()], user_id=7)
    assert type(tools[0]).__name__ == "UserBoundTool"
    assert type(tools[1]) is PlainTool  # 不含 user_id 的工具原样透传


def test_bind_user_forces_bound_id_despite_model_input():
    """模型传入 user_id=999（幻觉/注入），执行时必须被覆盖为绑定用户 7。"""
    wrapped = bind_user([EchoTool()], user_id=7)[0]
    result = json.loads(wrapped.invoke({"user_id": 999, "query": "hi"}))
    assert result["user_id"] == 7
    assert result["query"] == "hi"


def test_bind_user_forces_bound_id_async():
    wrapped = bind_user([EchoTool()], user_id=7)[0]
    result = json.loads(asyncio.run(wrapped._arun(user_id=888, query="async")))
    assert result["user_id"] == 7


def test_pretty_tool_name_strips_namespace():
    assert pretty_tool_name("course__get_my_schedule") == "get_my_schedule"
    assert pretty_tool_name("get_my_scores") == "get_my_scores"
