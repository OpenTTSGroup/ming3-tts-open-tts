from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache(maxsize=1)
def get_image_variant() -> Literal["0.5b", "tiny"]:
    marker = Path(__file__).resolve().parents[1] / "image-variant"
    variant = marker.read_text(encoding="utf-8").strip()
    if variant == "0.5b":
        return "0.5b"
    if variant == "tiny":
        return "tiny"
    raise ValueError(f"Invalid image variant in {marker}: {variant!r}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    ming3_model: str = Field(
        default_factory=lambda: f"inclusionAI/Ming-3.0-tts-{get_image_variant()}",
        description="Model repository or local checkpoint for this image variant.",
    )
    ming3_revision: str | None = None
    ming3_device: Literal["auto", "cuda", "cpu"] = "auto"
    ming3_cuda_index: int = Field(0, ge=0, le=255, description="CUDA device index.")
    ming3_dtype: Literal["bfloat16"] = "bfloat16"
    ming3_text_normalization: bool = False
    ming3_default_language: Literal["Chinese", "English", "German", "Spanish", "French", "Korean", "Japanese"] = "Chinese"
    ming3_default_voice: str = "灵小柔"
    ming3_seed: int = Field(1234, ge=0, le=4294967295, description="Generation seed.")
    ming3_max_decode_steps: int = Field(1000, ge=1, le=16384, description="Maximum decode steps per fragment.")
    ming3_max_length: int = Field(24, ge=1, le=512, description="Target text fragment length.")

    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535, description="HTTP port.")
    log_level: str = "info"
    voices_dir: str = "/voices"
    max_input_chars: int = Field(8000, ge=1, le=1000000, description="Input character limit.")
    default_response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    # Both upstream runtimes mutate shared generation state and process RNGs.
    max_concurrency: int = Field(1, ge=1, le=1, description="One inference per model instance.")
    max_queue_size: int = Field(0, ge=0, le=100000, description="Waiting requests; zero is unlimited.")
    queue_timeout: float = Field(0, ge=0, le=86400, allow_inf_nan=False, description="Queue timeout seconds; zero disables it.")
    max_audio_bytes: int = Field(20 * 1024 * 1024, ge=1, le=2147483647, description="Reference upload size limit.")
    cors_enabled: bool = False

    @property
    def image_variant(self) -> Literal["0.5b", "tiny"]:
        return get_image_variant()

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
