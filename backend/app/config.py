from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，可通过 .env 覆盖。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "course-agent"
    debug: bool = True

    # 开发默认使用本地 SQLite，后续可切换 MySQL
    database_url: str = "sqlite:///./course_agent.db"

    # JWT 认证（生产环境务必通过 .env 覆盖）
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # LLM（OpenAI 兼容接口，供应商可切换：千问 DashScope / DeepSeek / 其他）
    # 兼容旧环境变量名 LLM_* / DEEPSEEK_*（前者优先）
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    llm_model: str = Field(
        default="qwen-plus",
        validation_alias=AliasChoices("LLM_MODEL", "DEEPSEEK_MODEL"),
    )

    # 学期第一周的周一日期（用于推算当前教学周）
    semester_start: str = "2026-09-07"

    # Agent 行为调优（可通过 .env 覆盖）
    # 上下文预算：历史总字符超过 max 时，旧消息压缩为摘要，仅保留最近 keep 字符原文
    agent_history_max_chars: int = 8000
    agent_history_keep_recent_chars: int = 3000
    # 规划执行：单次任务最多拆解步骤数、单步 LLM/工具重试次数、失败后最多重规划次数
    agent_max_steps: int = 5
    agent_step_retries: int = 2
    agent_max_replans: int = 2
    # 写操作审批（human-in-the-loop）：命中的工具执行前须经 SSE 审批，超时视为拒绝
    agent_approval_tools: str = "save_study_plan"
    agent_approval_timeout: int = 60

    # 检索管线：混合召回（BM25 + 向量 RRF 融合）与交叉编码器重排
    rag_hybrid: bool = True
    rag_rerank: bool = True
    rag_recall_k: int = 20      # 每路召回候选数
    rag_rerank_top: int = 10    # 送入重排的候选数（重排后取 top_k）
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"


settings = Settings()
