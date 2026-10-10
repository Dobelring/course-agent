"""Token/成本核算：UsageCollector 提取、聚合与计价。"""
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.agent.usage import UsageCollector


def _result(usage: dict | None = None, usage_metadata=None) -> LLMResult:
    msg = AIMessage(content="", usage_metadata=usage_metadata)
    return LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output={"usage": usage} if usage else {},
    )


def test_extracts_openai_usage_format():
    c = UsageCollector(price_input_per_m=1.0, price_output_per_m=2.0)
    c.on_llm_end(_result({"prompt_tokens": 1000, "completion_tokens": 500}),
                 run_id=uuid4())
    assert c.input_tokens == 1000
    assert c.output_tokens == 500
    assert c.calls == 1
    # 1000/1e6×1 + 500/1e6×2 = 0.002
    assert abs(c.cost_cny - 0.002) < 1e-9


def test_extracts_usage_metadata_fallback():
    """无 llm_output.usage 时回退到 message.usage_metadata。"""
    c = UsageCollector(price_input_per_m=0.8, price_output_per_m=2.0)
    c.on_llm_end(_result(None, usage_metadata={
        "input_tokens": 200, "output_tokens": 100, "total_tokens": 300}),
        run_id=uuid4())
    assert c.input_tokens == 200
    assert c.output_tokens == 100


def test_aggregates_multiple_calls():
    """一次请求多次 LLM 调用（分类/规划/步骤/综合）累加。"""
    c = UsageCollector(price_input_per_m=1.0, price_output_per_m=1.0)
    for _ in range(3):
        c.on_llm_end(_result({"prompt_tokens": 100, "completion_tokens": 50}),
                     run_id=uuid4())
    assert c.calls == 3
    assert c.input_tokens == 300
    assert c.output_tokens == 150
    assert abs(c.cost_cny - 0.00045) < 1e-9


def test_zero_usage_not_counted():
    """部分网关不回 usage → 不计入 calls，避免虚增调用次数。"""
    c = UsageCollector()
    c.on_llm_end(_result(None), run_id=uuid4())
    assert c.calls == 0
    assert c.input_tokens == 0


def test_to_dict_shape():
    c = UsageCollector(price_input_per_m=1.0, price_output_per_m=2.0)
    c.on_llm_end(_result({"prompt_tokens": 1000, "completion_tokens": 500}),
                 run_id=uuid4())
    d = c.to_dict()
    assert set(d) == {"input_tokens", "output_tokens", "total_tokens", "calls", "cost_cny"}
    assert d["total_tokens"] == 1500
