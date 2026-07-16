from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    model: str = "claude-opus-4-7"
    max_tokens: int = 16000
    debug: bool = False


settings = Settings()
