from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL, make_url


BACKEND_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    app_name: str = "VANTA API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    mysql_user: str = "vanta"
    mysql_password: str = "change-me"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3308
    mysql_database: str = "vanta"
    database_url_override: str | None = Field(default=None, validation_alias="DATABASE_URL")
    prediction_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    prediction_max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    model_config = SettingsConfigDict(
        env_file=(REPOSITORY_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def database_url(self) -> URL:
        if self.database_url_override:
            return make_url(self.database_url_override)

        return URL.create(
            drivername="mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
