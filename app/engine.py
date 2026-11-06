from __future__ import annotations

import json
import logging
import random
import sys
import tempfile
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)
LANGUAGES = ("Chinese", "English", "German", "Spanish", "French", "Korean", "Japanese")
BUILTIN_VOICES: tuple[str, ...] = tuple(json.loads(
    Path(__file__).with_name("builtin_voices.json").read_text(encoding="utf-8")
))


class TTSEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        root = Path(__file__).resolve().parents[1] / "engine"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import torch
        from huggingface_hub import snapshot_download

        if settings.ming3_device == "cpu" or not torch.cuda.is_available():
            raise RuntimeError("This adapter supports the upstream CUDA/bfloat16 runtime; CPU is not supported")
        self.device = f"cuda:{settings.ming3_cuda_index}"
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Ming-3.0-tts requires bfloat16 GPU support")
        self.dtype_str = "bfloat16"
        self.quantization = settings.ming3_quantization
        if settings.ming3_default_voice not in BUILTIN_VOICES:
            raise ValueError("MING3_DEFAULT_VOICE is not in the built-in catalog")
        path = Path(settings.ming3_model).expanduser()
        if not path.is_dir():
            path = Path(snapshot_download(repo_id=settings.ming3_model, revision=settings.ming3_revision))
        if settings.image_variant == "tiny":
            from ming_tiny import MingTinyTTS
            loading = nullcontext()
            if self.quantization != "none":
                from app.quantization import quantized_tiny_loading
                loading = quantized_tiny_loading(self.quantization)
            with loading:
                self.model = MingTinyTTS.from_pretrained(
                    str(path), device=self.device, dtype="bf16", moe_backend="loop",
                    text_normalization=settings.ming3_text_normalization,
                )
        else:
            from ming_tts import MingTTS
            self.model = MingTTS(str(path), device=self.device,
                                 text_normalization=settings.ming3_text_normalization,
                                 cache_prompts=False)
        self.sample_rate = int(self.model.sample_rate)

    def list_languages(self) -> list[tuple[str, str]]:
        return [(key, key) for key in LANGUAGES]

    def _generate(self, task: str, text: str, *, refs: list[str] | None = None,
                  transcripts: list[str] | None = None, language: str | None = None,
                  controls: dict | None = None, instruction: str | None = None,
                  seed: int | None = None, cfg_scale: float | None = None,
                  sigma: float | None = None, temperature: float | None = None) -> np.ndarray:
        import torch

        torch.cuda.set_device(self.device)
        seed = self.settings.ming3_seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        common: dict[str, Any] = dict(
            text=text, sigma=0.25 if sigma is None else sigma,
            temperature=0.0 if temperature is None else temperature,
            max_length=self.settings.ming3_max_length,
        )
        scale = 2.0 if cfg_scale is None else cfg_scale
        # Decode all upload formats via FFmpeg/PyAV before the upstream
        # soundfile-only reader. Files are private and removed after inference.
        with tempfile.TemporaryDirectory(prefix="ming3-audio-") as directory:
            normalized = [self._reference(path, Path(directory) / f"{i}.wav")
                          for i, path in enumerate(refs or [])]
            prompt = normalized if task == "podcast" else (normalized[0] if normalized else None)
            transcript = transcripts if task == "podcast" else (transcripts[0] if transcripts else None)
            with torch.inference_mode():
                if self.settings.image_variant == "tiny":
                    kwargs = dict(controls or {})
                    if task == "dialect":
                        kwargs["dialect"] = {"cantonese": "广粤话", "chuanyu dialect": "川渝话"}[kwargs["dialect"]]
                    if task != "podcast":
                        kwargs["max_decode_steps"] = self.settings.ming3_max_decode_steps
                    output = self.model.generate(
                        task=task, prompt_wav=prompt, prompt_text=transcript,
                        target_language=language, instruction=instruction,
                        seed=seed, cfg_scale=scale, **common, **kwargs,
                    ).waveform
                else:
                    mapped = {"seed": "TTS", "multilingual": "MULTILINGUAL",
                              "voice_design": "VOICE_DESIGN", "freeform": "FREEFORM",
                              "ip": "IP", "emotion": "EMOTION", "dialect": "DIALECT", "basic": "BASIC"}[task]
                    output = self.model.synthesize(
                        task=mapped, prompt_wav_path=prompt, prompt_text=transcript,
                        language_id=language, instruction=instruction, controls=controls,
                        use_spk_emb=bool(normalized), use_zero_spk_emb=not normalized,
                        cfg=scale, max_decode_steps=self.settings.ming3_max_decode_steps, **common,
                    )
            samples = output.detach().float().cpu().numpy().reshape(-1)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise RuntimeError("upstream returned empty or non-finite audio")
        return np.clip(samples, -1, 1).astype(np.float32, copy=False)

    @staticmethod
    def _reference(source: str, target: Path) -> str:
        import av
        import soundfile as sf

        with av.open(source) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
            with sf.SoundFile(str(target), "w", samplerate=24000, channels=1,
                              format="WAV", subtype="FLOAT") as dest:
                for frame in container.decode(stream):
                    for converted in resampler.resample(frame):
                        dest.write(converted.to_ndarray().reshape(-1))
                for converted in resampler.resample(None):
                    dest.write(converted.to_ndarray().reshape(-1))
                if dest.tell() == 0:
                    raise ValueError("reference audio is empty")
        return str(target)

    async def _run(self, task: str, text: str, **kwargs: Any) -> np.ndarray:
        # Do not abandon the worker on disconnect: retain the queue permit and
        # uploaded files until GPU work ends (asyncio.to_thread alone cannot).
        with anyio.CancelScope(shield=True):
            return await anyio.to_thread.run_sync(partial(self._generate, task, text, **kwargs))

    async def synthesize_clone(self, text: str, *, ref_audio: str, ref_text: str,
                               ref_mtime: float | None = None, instructions: str | None = None,
                               speed: float = 1.0, language: str | None = None,
                               emotion: str | None = None, dialect: str | None = None,
                               attribute_speed: str | None = None, pitch: str | None = None,
                               volume: str | None = None, **options: Any) -> np.ndarray:
        self._ignored(speed, instructions)
        controls = None
        if emotion:
            task, controls = "emotion", {"emotion": emotion}
        elif dialect:
            task, controls = "dialect", {"dialect": dialect}
        elif any(v is not None for v in (attribute_speed, pitch, volume)):
            task, controls = "basic", dict(speed=attribute_speed or "中速", pitch=pitch or "中", volume=volume or "中")
        else:
            language = language or self.settings.ming3_default_language
            task = "seed" if language in ("Chinese", "English") else "multilingual"
        if controls:
            language = None
        return await self._run(task, text, refs=[ref_audio],
                               transcripts=None if controls else [ref_text],
                               language=language, controls=controls, **options)

    @staticmethod
    def _ignored(speed: float, instructions: str | None) -> None:
        if speed != 1.0:
            log.warning("numeric speed ignored; use attribute_speed for BASIC control")
        if instructions:
            log.warning("instructions ignored; use /v1/audio/design with mode=freeform")

    async def synthesize_builtin(self, text: str, *, voice: str,
                                 instructions: str | None = None, speed: float = 1.0,
                                 **options: Any) -> np.ndarray:
        self._ignored(speed, instructions)
        return await self._run("ip", text, controls={"ip": voice}, **options)

    async def synthesize_design(self, text: str, *, instruct: str | None,
                                mode: str | None = None, **options: Any) -> np.ndarray:
        description = (instruct or "").strip()
        if not description:
            return await self.synthesize_builtin(text, voice=self.settings.ming3_default_voice, **options)
        if mode == "freeform":
            return await self._run("freeform", text, instruction=description, **options)
        return await self._run("voice_design", text, controls={"style": description}, **options)

    async def synthesize_dialogue(self, text: str, *, refs: list[str],
                                  transcripts: list[str], **options: Any) -> np.ndarray:
        return await self._run("podcast", text, refs=refs, transcripts=transcripts, **options)
