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


settings = Settings()
