from pydantic_settings import BaseSettings, SettingsConfigDict

# The three unified Qwen3.7 models the service exposes. Users may pick any of
# these per request; anything else falls back to `default_model`.
ALLOWED_MODELS: frozenset[str] = frozenset(
    {"qwen3.7-max", "qwen3.7-plus", "qwen3.7-flash"}
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    dashscope_api_key: str = ""
    llm_base_url: str = (
        "https://llm-6qoqdsrsqplw1w8w.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    model: str = "qwen3.7-flash"          # 默认主模型（快速）
    default_model: str = "qwen3.7-flash"  # 非法 model 时回落
    fallback_model: str = "qwen3.7-plus"  # 主模型过载/context 超长时降级
    max_tokens: int = 5000
    enable_thinking: bool = False         # qwen3.7 深度思考（默认关闭）
    debug: bool = False

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "agent_release"

    def resolve_model(self, name: str | None) -> str:
        """Validate a requested model name, falling back to the default."""
        if name and name in ALLOWED_MODELS:
            return name
        return self.default_model


settings = Settings()
