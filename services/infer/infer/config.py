"""Application settings — read from env vars or CLI flags."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All values can be overridden via environment variables prefixed with INFER_."""

    model_path: str = Field(
        default="models/DeepSeek-R1-Distill-Qwen-1.5B",
        description="Path to the model directory (relative to project root or absolute).",
    )
    device: str = Field(
        default="cpu",
        description="Device to run inference on: cpu | nvidia",
    )
    max_ctx_len: int = Field(
        default=2048,
        description="Maximum context length for KV cache allocation.",
    )
    max_batch_size: int = Field(
        default=1,
        description=(
            "Maximum number of requests processed per batch iteration. "
            "Keep at 1 until the C backend supports per-request KV caches."
        ),
    )
    max_pool_size: int = Field(
        default=128,
        description="Maximum number of pending requests in the pool.",
    )
    host: str = Field(default="0.0.0.0", description="Bind host.")
    port: int = Field(default=8000, description="Bind port.")
    model_name: str = Field(
        default="deepseek-r1-distill-qwen-1.5b",
        description="Model name exposed via /v1/models and accepted in requests.",
    )

    model_config = {"env_prefix": "INFER_"}


settings = Settings()
