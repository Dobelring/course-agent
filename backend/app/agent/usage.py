"""Token 用量与成本核算：LangChain 回调收集，按请求聚合。

收集点：build_llm(callbacks=[collector]) 构造的 ChatOpenAI 实例上挂回调，
该实例被 classify / plan / agent 步骤 / 综合所有调用复用 → 一次请求一个
collector 覆盖全部 LLM 调用（含 astream_events 内部的步骤调用）。

计费口径：input/output token 分开累计，成本 = Σ(token/1e6 × 单价)，
单价来自 Settings（按所用模型官网价配置），落库到 AiMessage 供仪表盘聚合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class UsageCollector(BaseCallbackHandler):
    """聚合单次请求内所有 LLM 调用的 token 用量（asyncio 单线程，无需加锁）。"""

    price_input_per_m: float = 0.0   # 元 / 百万 input token
    price_output_per_m: float = 0.0  # 元 / 百万 output token
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_cny: float = field(default=0.0)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs) -> None:
        in_tok, out_tok = self._extract(response)
        if in_tok == 0 and out_tok == 0:
            return
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cost_cny = round(
            self.input_tokens / 1e6 * self.price_input_per_m
            + self.output_tokens / 1e6 * self.price_output_per_m,
            6,
        )

    @staticmethod
    def _extract(response: LLMResult) -> tuple[int, int]:
        """从 LLMResult 提取 token 数，兼容两种载体：
        1) llm_output["usage"]（OpenAI 原生字段，流式含 include_usage 时也有）
        2) generation.message.usage_metadata（langchain 统一的 UsageMetadata）
        """
        usage = (response.llm_output or {}).get("usage") or \
                (response.llm_output or {}).get("token_usage") or {}
        if usage:
            in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            return in_tok, out_tok

        in_tok = out_tok = 0
        for generations in response.generations:
            for gen in generations:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if not meta:
                    continue
                # usage_metadata 是 dict（langchain 统一格式），兼容属性访问
                if isinstance(meta, dict):
                    in_tok += int(meta.get("input_tokens") or 0)
                    out_tok += int(meta.get("output_tokens") or 0)
                else:
                    in_tok += int(getattr(meta, "input_tokens", 0) or 0)
                    out_tok += int(getattr(meta, "output_tokens", 0) or 0)
        return in_tok, out_tok

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "calls": self.calls,
            "cost_cny": round(self.cost_cny, 6),
        }
