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
    redis_cache_similarity_threshold: float = Field(default=0.95, alias="REDIS_CACHE_SIMILARITY_THRESHOLD")
    redis_cache_max_entries: int = Field(default=2000, alias="REDIS_CACHE_MAX_ENTRIES")

    mysql_host: str = Field(default="127.0.0.1", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_db: str = Field(default="cp_rag_db", alias="MYSQL_DB")
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
    milvus_vector_field: str = Field(default="embedding", alias="MILVUS_VECTOR_FIELD")
    milvus_text_field: str = Field(default="context", alias="MILVUS_TEXT_FIELD")
    milvus_output_fields: str = Field(
        default="id,title,context,source",
        alias="MILVUS_OUTPUT_FIELDS",
        description="Comma-separated output fields for Milvus search results.",
    )
    rag_top_k: int = Field(default=2, alias="RAG_TOP_K")
    embedding_model_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL_NAME")
    llm_provider: str = Field(default="openai_compatible", alias="LLM_PROVIDER")
    llm_api_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_API_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")

    v2_api_prefix: str = Field(default="/api/v2", alias="V2_API_PREFIX")
    v2_milvus_collection: str = Field(default="cp_rag_document_v2", alias="V2_MILVUS_COLLECTION")
    v2_parent_chunk_size: int = Field(default=512, alias="V2_PARENT_CHUNK_SIZE")
    v2_parent_chunk_overlap: int = Field(default=64, alias="V2_PARENT_CHUNK_OVERLAP")
    v2_child_chunk_size: int = Field(default=128, alias="V2_CHILD_CHUNK_SIZE")
    v2_child_chunk_overlap: int = Field(default=32, alias="V2_CHILD_CHUNK_OVERLAP")
    v2_rag_top_k: int = Field(default=4, alias="V2_RAG_TOP_K")
    v2_qwen_api_base_url: str = Field(default="https://api.openai.com/v1", alias="V2_QWEN_API_BASE_URL")
    v2_qwen_api_key: str = Field(default="", alias="V2_QWEN_API_KEY")
    v2_qwen_text_model: str = Field(default="qwen-plus", alias="V2_QWEN_TEXT_MODEL")
    v2_qwen_vision_model: str = Field(default="qwen-vl-plus", alias="V2_QWEN_VISION_MODEL")
    v2_bert_classifier_model_path: str = Field(
        default=r"D:\S\heimatool\RAG\integrated_qa_system\rag_qa\models\bert_query_classifier",
        alias="V2_BERT_CLASSIFIER_MODEL_PATH",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
