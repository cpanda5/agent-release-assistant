from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    dashscope_api_key: str = ""
    model: str = "qwen3-6-flash"          # 主模型（当前：免费快速推理）
    # model: str = "qwen-plus"            # 可选：更强，适合复杂分析
    # model: str = "qwen-max"             # 可选：最强，适合代码生成
    fallback_model: str = "qwen-turbo"    # 降级模型（主模型过载或 context 超长时切换）
    max_tokens: int = 5000
    debug: bool = False

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "agent_release"


settings = Settings()
