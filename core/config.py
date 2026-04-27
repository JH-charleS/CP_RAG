from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="CP_RAG", alias="APP_NAME")
    app_env: Literal["local", "dev", "test", "prod"] = Field(default="local", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")

    redis_host: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    redis_prefix: str = Field(default="cp_rag", alias="REDIS_PREFIX")
    redis_max_connections: int = Field(default=100, alias="REDIS_MAX_CONNECTIONS")

    mysql_host: str = Field(default="127.0.0.1", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_db: str = Field(default="cp_rag", alias="MYSQL_DB")
    mysql_charset: str = Field(default="utf8mb4", alias="MYSQL_CHARSET")
    mysql_min_size: int = Field(default=1, alias="MYSQL_MIN_SIZE")
    mysql_max_size: int = Field(default=10, alias="MYSQL_MAX_SIZE")
    mysql_connect_timeout: int = Field(default=10, alias="MYSQL_CONNECT_TIMEOUT")
    mysql_autocommit: bool = Field(default=False, alias="MYSQL_AUTOCOMMIT")

    milvus_host: str = Field(default="127.0.0.1", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_user: str = Field(default="", alias="MILVUS_USER")
    milvus_password: str = Field(default="", alias="MILVUS_PASSWORD")
    milvus_db_name: str = Field(default="default", alias="MILVUS_DB_NAME")
    milvus_timeout: float = Field(default=10.0, alias="MILVUS_TIMEOUT")
    milvus_secure: bool = Field(default=False, alias="MILVUS_SECURE")
    milvus_collection: str = Field(default="cp_rag_solutions", alias="MILVUS_COLLECTION")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
