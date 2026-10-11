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
    # 计价（元/百万 token）：按所用模型官网价配置，用于成本核算展示
    llm_price_input_per_m: float = Field(
        default=0.8, validation_alias=AliasChoices("LLM_PRICE_INPUT_PER_M"),
    )
    llm_price_output_per_m: float = Field(
        default=2.0, validation_alias=AliasChoices("LLM_PRICE_OUTPUT_PER_M"),
    )
    # 多模型路由（主备）：JSON 数组，按 priority 升序为主备顺序，同级按 weight 加权分流。
    # 留空则用上面 llm_api_key/llm_base_url/llm_model 构造单路由（向后兼容）。
    # 每项：{"name","base_url","model","api_key"(可空,回退 llm_api_key),"priority","weight"}
    llm_routes: str = Field(default="", validation_alias=AliasChoices("LLM_ROUTES"))
    # 熔断：同一路由连续失败 N 次后熔断（跳过该路由），冷却 cooldown 秒后半开探测
    llm_breaker_threshold: int = 2
    llm_breaker_cooldown_s: int = 30

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
    # 三态审批：单步最多「修改后执行」轮数（用户给修改意见 → 模型调整后重新发起审批）
    agent_approval_max_revises: int = 2

    # 反思环节：综合产出后自检质量（1-5 分），低于阈值且给出修订则替换答案
    agent_reflect: bool = True
    agent_reflect_min_quality: int = 3

    # 工具幂等：save_study_plan 按 (user, 内容指纹) 在窗口期内去重，窗口内重复保存返回既有计划
    tool_idempotency_window_s: int = 600

    # MCP 工具列表缓存 TTL（秒）：spawn+列举实测 ~1.5s/请求，缓存后降为一次性成本；0 = 关闭
    mcp_tools_cache_ttl: int = 300

    # 限流（每分钟）：聊天按用户、登录按客户端 IP；0 = 关闭（测试/评测脚本）
    rate_limit_chat_per_min: int = 10
    rate_limit_login_per_min: int = 10

    # 跨会话画像：每收到 N 条助手消息刷新一次用户画像（LLM 合并）
    memory_refresh_every: int = 4

    # 检索管线：混合召回（BM25 + 向量 RRF 融合）与交叉编码器重排
    rag_hybrid: bool = True
    rag_rerank: bool = True
    rag_recall_k: int = 20      # 每路召回候选数
    rag_rerank_top: int = 10    # 送入重排的候选数（重排后取 top_k）
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"
    rag_rerank_batch: int = 32  # 交叉编码分批大小（控制内存峰值）
    # 启动预热：lifespan 里加载 embedding/重排模型并跑一次空推理（best-effort）
    rag_warmup: bool = True
    # 切片：递归切分目标长度与相邻块重叠（重叠保留跨块上下文）
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 75
    # 检索结果多样性：同一文档最多贡献的切片数、相邻切片合并
    rag_max_per_doc: int = 2


settings = Settings()
