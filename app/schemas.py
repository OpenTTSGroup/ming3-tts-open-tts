# Keep multipart dependency annotations concrete for FastAPI class inspection.

from typing import Any, Literal

from fastapi import File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
LanguageKey = Literal["Chinese", "English", "German", "Spanish", "French", "Korean", "Japanese"]
Emotion = Literal["高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性"]
Dialect = Literal["cantonese", "chuanyu dialect"]
AttributeSpeed = Literal["慢速", "中速", "快速"]
Level = Literal["低", "中", "高"]


class Capabilities(BaseModel):
    clone: bool = Field(description="Reference voice cloning.")
    streaming: bool = Field(description="Incremental audio generation.")
    design: bool = Field(description="Text-only voice design.")
    languages: bool = Field(description="Explicit language discovery.")
    builtin_voices: bool = Field(description="Built-in IP voices.")


class ConcurrencySnapshot(BaseModel):
    max: int = Field(ge=1, description="Concurrency ceiling.")
    active: int = Field(ge=0, description="Active inference jobs.")
    queued: int = Field(ge=0, description="Waiting requests.")


class HealthResponse(BaseModel):
    status: Literal["loading", "ok"] = Field(description="Model readiness.")
    model: str = Field(description="Configured model ID or path.")
    sample_rate: int = Field(ge=0, description="Native output rate; zero while loading.")
    capabilities: Capabilities = Field(description="API capabilities.")
    device: str | None = Field(None, description="Actual device.")
    dtype: str | None = Field(None, description="Floating-point compute dtype; quantization reports packed expert storage.")
    quantization: Literal["none", "int8", "int4"] = Field(
        "none", description="MoE expert weight storage; floating-point computation remains bfloat16.",
    )
    concurrency: ConcurrencySnapshot | None = Field(None, description="Queue status.")


class VoiceInfo(BaseModel):
    id: str = Field(description="Built-in ID or file://<id>.")
    preview_url: str | None = Field(None, description="File preview URL; null for built-ins.")
    prompt_text: str | None = Field(None, description="Reference transcript; null for built-ins.")
    metadata: dict[str, Any] | None = Field(None, description="Optional YAML metadata.")


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo] = Field(description="Available voices.")


class Language(BaseModel):
    key: str = Field(description="Native language identifier.")
    name: str = Field(description="Display name.")


class LanguagesResponse(BaseModel):
    languages: list[Language] = Field(description="Supported clone target languages.")


class GenerationOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    seed: int | None = Field(None, ge=0, le=4294967295, description="Generation seed; defaults to MING3_SEED.")
    cfg_scale: float | None = Field(None, ge=0, le=10, allow_inf_nan=False, description="Guidance scale; default 2.")
    temperature: float | None = Field(None, ge=0, le=2, allow_inf_nan=False, description="Sampling temperature; default 0.")
    sigma: float | None = Field(None, ge=0, le=1, allow_inf_nan=False, description="Flow noise scale; default 0.25.")


class CloneOptions(GenerationOptions):
    language: LanguageKey | None = Field(None, description="Clone target language; defaults to MING3_DEFAULT_LANGUAGE.")
    emotion: Emotion | None = Field(None, description="Speaker-only emotion task; exclusive with other controls.")
    dialect: Dialect | None = Field(None, description="Speaker-only dialect task; exclusive with other controls.")
    attribute_speed: AttributeSpeed | None = Field(None, description="BASIC speed; omitted BASIC attributes default to medium.")
    pitch: Level | None = Field(None, description="BASIC pitch; low, medium or high.")
    volume: Level | None = Field(None, description="BASIC volume; low, medium or high.")

    @model_validator(mode="after")
    def exclusive_controls(self) -> "CloneOptions":
        basic = any(v is not None for v in (self.attribute_speed, self.pitch, self.volume))
        if sum((self.emotion is not None, self.dialect is not None, basic)) > 1:
            raise ValueError("emotion, dialect and BASIC attributes are separate tasks")
        return self


class SpeechRequest(CloneOptions):
    model: str | None = Field(None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(min_length=1, description="Text to synthesize; inline paralinguistic tags are preserved.")
    voice: str = Field(min_length=1, description="Built-in voice ID or file://<id>.")
    response_format: ResponseFormat | None = Field(None, description="Output format; service default is mp3.")
    speed: float = Field(1, ge=0.25, le=4, allow_inf_nan=False, description="Accepted but ignored; use attribute_speed for BASIC control.")
    instructions: str | None = Field(None, description="Accepted but ignored; use design mode=freeform for instructions.")


class DesignRequest(GenerationOptions):
    input: str = Field(min_length=1, description="Text to synthesize.")
    instruct: str | None = Field(None, description="Voice description or freeform instruction; blank selects default built-in voice.")
    response_format: ResponseFormat | None = Field(None, description="Output format; service default is mp3.")
    mode: Literal["voice_design", "freeform"] | None = Field(None, description="Design task; default voice_design.")


class DialogueRequest(GenerationOptions):
    input: str = Field(min_length=1, description="One speaker_N: utterance per line, with contiguous speaker IDs starting at 1.")
    voices: list[str] = Field(min_length=2, max_length=16, description="Ordered file:// references matching speaker_N; Tiny only.")
    response_format: ResponseFormat | None = Field(None, description="Output format; service default is mp3.")


class CloneRequest:
    """Multipart dependency shared by OpenAPI and upload validation."""

    def __init__(
        self, audio: UploadFile = File(..., description="Reference audio upload."),
        prompt_text: str = Form(..., description="Required reference transcript; content is ignored for controlled speaker-only tasks."),
        input: str = Form(..., description="Text to synthesize."),
        response_format: ResponseFormat | None = Form(None, description="Output format; service default is mp3."),
        speed: float = Form(1, ge=0.25, le=4, allow_inf_nan=False, description="Accepted but ignored."),
        instructions: str | None = Form(None, description="Accepted but ignored; use design mode=freeform."),
        model: str | None = Form(None, description="Accepted for OpenAI compatibility; ignored."),
        language: LanguageKey | None = Form(None, description="Clone target language."),
        emotion: Emotion | None = Form(None, description="Emotion control."),
        dialect: Dialect | None = Form(None, description="Dialect control."),
        attribute_speed: AttributeSpeed | None = Form(None, description="BASIC speed control."),
        pitch: Level | None = Form(None, description="BASIC pitch control."),
        volume: Level | None = Form(None, description="BASIC volume control."),
        seed: int | None = Form(None, ge=0, le=4294967295, description="Generation seed."),
        cfg_scale: float | None = Form(None, ge=0, le=10, allow_inf_nan=False, description="Guidance scale."),
        temperature: float | None = Form(None, ge=0, le=2, allow_inf_nan=False, description="Temperature."),
        sigma: float | None = Form(None, ge=0, le=1, allow_inf_nan=False, description="Flow noise scale."),
    ) -> None:
        self.audio, self.prompt_text, self.input = audio, prompt_text, input
        self.response_format, self.speed, self.instructions = response_format, speed, instructions
        try:
            self.options = CloneOptions(language=language, emotion=emotion, dialect=dialect,
                                        attribute_speed=attribute_speed, pitch=pitch, volume=volume,
                                        seed=seed, cfg_scale=cfg_scale, temperature=temperature, sigma=sigma)
        except ValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
