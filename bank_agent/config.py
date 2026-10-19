from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")
    demo_mode: bool = True
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "Qwen/Qwen3-8B"
    llm_revision: str = "operator-configured"
    llm_structured_method: Literal["json_mode", "json_schema", "function_calling"] = "json_mode"
    llm_thinking: Literal["auto", "enabled", "disabled"] = "auto"
    tokenizer_model: str = ""
    tokenizer_revision: str = "main"
    embedding_provider: Literal["openai_compatible", "local"] = "openai_compatible"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = "http://localhost:8002/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "main"
    reranker_provider: Literal["lexical", "http", "local"] = "http"
    reranker_api_key: SecretStr = SecretStr("")
    reranker_url: str = "http://localhost:8003/v1/rerank"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "main"
    api_auth_tokens_json: SecretStr = SecretStr("")
    request_timeout: float = Field(default=60, gt=0, le=300)
    chunk_size: int = Field(default=600, ge=100, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=300)
    retrieval_k: int = Field(default=20, ge=1, le=100)
    rerank_k: int = Field(default=30, ge=1, le=100)
    evidence_k: int = Field(default=8, ge=1, le=20)
    max_supplement_rounds: int = Field(default=2, ge=0, le=2)
    max_subquestions: int = Field(default=4, ge=1, le=4)
    max_context_tokens: int = Field(default=6000, ge=500, le=32000)
    max_output_tokens: int = Field(default=1500, ge=100, le=8000)
    api_url: str = "http://127.0.0.1:8000"

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
