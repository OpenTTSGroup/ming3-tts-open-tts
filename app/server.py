from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import anyio
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from app.audio import encode
from app.concurrency import ConcurrencyLimiter
from app.config import get_settings
from app.engine import BUILTIN_VOICES, LANGUAGES, TTSEngine
from app.schemas import (
    Capabilities, CloneOptions, CloneRequest, DesignRequest, DialogueRequest,
    GenerationOptions, HealthResponse, Language, LanguagesResponse,
    SpeechRequest, VoiceInfo, VoiceListResponse,
)
from app.voices import FILE_VOICE_PREFIX, Voice, VoiceCatalog

log = logging.getLogger(__name__)
CAPABILITIES = Capabilities(clone=True, streaming=False, design=True,
                            languages=True, builtin_voices=True)
CLONE_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app.state.settings = settings
    app.state.catalog = VoiceCatalog(settings.voices_path)
    app.state.limiter = ConcurrencyLimiter(settings.max_concurrency, settings.max_queue_size, settings.queue_timeout)
    app.state.engine = None

    async def initialize() -> None:
        try:
            app.state.engine = await asyncio.to_thread(TTSEngine, settings)
            log.info("Ming ready: variant=%s model=%s device=%s", settings.image_variant,
                     settings.ming3_model, app.state.engine.device)
        except Exception:
            log.exception("Ming model loading failed; exiting")
            logging.shutdown()
            os._exit(1)

    task = asyncio.create_task(initialize())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="ming3-tts-open-tts", version="1.0.0", lifespan=lifespan)
if get_settings().cors_enabled:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _engine(request: Request) -> TTSEngine:
    if request.app.state.engine is None:
        raise HTTPException(503, "engine loading")
    return request.app.state.engine


def _validate_text(text: str, limit: int) -> None:
    if not text.strip():
        raise HTTPException(422, "input must not be empty")
    if len(text) > limit:
        raise HTTPException(413, f"input exceeds {limit} chars")


def _voice(voice: str, request: Request) -> Voice | str:
    if voice.startswith(FILE_VOICE_PREFIX):
        found = request.app.state.catalog.get(voice)
        if found is None:
            raise HTTPException(404, f"voice '{voice}' not found")
        return found
    if "://" in voice:
        raise HTTPException(501, "remote voice URIs not supported")
    if voice not in BUILTIN_VOICES:
        raise HTTPException(404, f"voice '{voice}' not found")
    return voice


def _generation(req: GenerationOptions) -> dict:
    return {key: getattr(req, key) for key in GenerationOptions.model_fields}


def _clone_options(req: CloneOptions) -> dict:
    return {key: getattr(req, key) for key in CloneOptions.model_fields}


def _controlled(req: CloneOptions) -> bool:
    return any(getattr(req, key) is not None for key in ("emotion", "dialect", "attribute_speed", "pitch", "volume"))


async def _render(job: Awaitable[np.ndarray], rate: int, fmt: str) -> Response:
    try:
        samples = await job
    except Exception as exc:
        log.exception("inference failed")
        raise HTTPException(500, "inference failed") from exc
    try:
        body, mime = await anyio.to_thread.run_sync(encode, samples, rate, fmt)
    except Exception as exc:
        log.exception("audio encoding failed")
        raise HTTPException(500, "audio encoding failed") from exc
    return Response(body, media_type=mime)


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    engine = request.app.state.engine
    return HealthResponse(status="ok" if engine else "loading",
                          model=request.app.state.settings.ming3_model,
                          sample_rate=engine.sample_rate if engine else 0,
                          device=engine.device if engine else None,
                          dtype=engine.dtype_str if engine else None,
                          quantization=engine.quantization if engine else request.app.state.settings.ming3_quantization,
                          capabilities=CAPABILITIES,
                          concurrency=request.app.state.limiter.snapshot())


@app.get("/v1/audio/voices", response_model=VoiceListResponse)
async def voices(request: Request) -> VoiceListResponse:
    files = await anyio.to_thread.run_sync(request.app.state.catalog.list)
    return VoiceListResponse(voices=[VoiceInfo(id=name) for name in BUILTIN_VOICES] + [
        VoiceInfo(id=v.uri, preview_url=f"/v1/audio/voices/preview?id={quote(v.id, safe='')}",
                  prompt_text=v.prompt_text, metadata=v.metadata) for v in files
    ])


@app.get("/v1/audio/voices/preview")
async def preview(id: str, request: Request) -> FileResponse:
    if id.startswith(FILE_VOICE_PREFIX):
        raise HTTPException(404, "preview id must be a plain voice name")
    voice = await anyio.to_thread.run_sync(request.app.state.catalog.get, id)
    if voice is None:
        raise HTTPException(404, "voice not found")
    return FileResponse(voice.wav_path, media_type="audio/wav", filename=f"{voice.id}.wav")


@app.get("/v1/audio/languages", response_model=LanguagesResponse)
async def languages() -> LanguagesResponse:
    return LanguagesResponse(languages=[Language(key=key, name=key) for key in LANGUAGES])


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request) -> Response:
    settings = request.app.state.settings
    _validate_text(req.input, settings.max_input_chars)
    voice = await anyio.to_thread.run_sync(_voice, req.voice, request)
    if isinstance(voice, str) and _controlled(req):
        raise HTTPException(422, "emotion/dialect/BASIC controls require a file:// voice")
    engine = _engine(request)
    async with request.app.state.limiter.acquire():
        if isinstance(voice, Voice):
            job = engine.synthesize_clone(req.input, ref_audio=str(voice.wav_path),
                                          ref_text=voice.prompt_text, ref_mtime=voice.mtime,
                                          speed=req.speed, instructions=req.instructions, **_clone_options(req))
        else:
            job = engine.synthesize_builtin(req.input, voice=voice, speed=req.speed,
                                            instructions=req.instructions, **_generation(req))
        return await _render(job, engine.sample_rate, req.response_format or settings.default_response_format)


@app.post("/v1/audio/clone")
async def clone(request: Request, req: CloneRequest = Depends()) -> Response:
    settings = request.app.state.settings
    tmp: Path | None = None
    try:
        _validate_text(req.input, settings.max_input_chars)
        if not req.prompt_text.strip() and not _controlled(req.options):
            raise HTTPException(422, "prompt_text must not be empty for ordinary cloning")
        suffix = Path(req.audio.filename or "reference.wav").suffix.lower()
        if suffix not in CLONE_AUDIO_EXTS:
            raise HTTPException(415, "unsupported reference audio format")
        engine = _engine(request)
        with tempfile.NamedTemporaryFile(prefix="ming3-ref-", suffix=suffix, delete=False) as dest:
            tmp = Path(dest.name)
            size = 0
            while chunk := await req.audio.read(1 << 20):
                size += len(chunk)
                if size > settings.max_audio_bytes:
                    raise HTTPException(413, "reference audio exceeds MAX_AUDIO_BYTES")
                await anyio.to_thread.run_sync(dest.write, chunk)
        if size == 0:
            raise HTTPException(400, "audio file is empty")
        async with request.app.state.limiter.acquire():
            return await _render(engine.synthesize_clone(
                req.input, ref_audio=str(tmp), ref_text=req.prompt_text,
                instructions=req.instructions, speed=req.speed, **_clone_options(req.options),
            ), engine.sample_rate, req.response_format or settings.default_response_format)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        with anyio.CancelScope(shield=True):
            await req.audio.close()


@app.post("/v1/audio/design")
async def design(req: DesignRequest, request: Request) -> Response:
    settings = request.app.state.settings
    _validate_text(req.input, settings.max_input_chars)
    engine = _engine(request)
    async with request.app.state.limiter.acquire():
        return await _render(engine.synthesize_design(
            req.input, instruct=req.instruct, mode=req.mode, **_generation(req),
        ), engine.sample_rate, req.response_format or settings.default_response_format)


if get_settings().image_variant == "tiny":
    @app.post("/v1/audio/dialogue")
    async def dialogue(req: DialogueRequest, request: Request) -> Response:
        settings = request.app.state.settings
        _validate_text(req.input, settings.max_input_chars)
        speakers = set()
        for line in req.input.splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"\s*speaker_([1-9]\d*):\s*(\S.*)", line)
            if match is None:
                raise HTTPException(422, "each dialogue line must be speaker_N: text")
            speakers.add(int(match.group(1)))
        if speakers != set(range(1, len(req.voices) + 1)):
            raise HTTPException(422, "speaker IDs must match the ordered voices list")
        references = []
        for vid in req.voices:
            if not vid.startswith(FILE_VOICE_PREFIX):
                raise HTTPException(422, "dialogue requires file:// voices")
            references.append(await anyio.to_thread.run_sync(_voice, vid, request))
        engine = _engine(request)
        async with request.app.state.limiter.acquire():
            return await _render(engine.synthesize_dialogue(
                req.input, refs=[str(v.wav_path) for v in references],
                transcripts=[v.prompt_text for v in references], **_generation(req),
            ), engine.sample_rate, req.response_format or settings.default_response_format)
