# ming3-tts-open-tts

**English** · [中文](./README.zh.md)

An [Open TTS](https://github.com/OpenTTSGroup/open-tts-spec) adapter for [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts). Supports 0.5B and Tiny with file/upload cloning, 241 built-in voices, seven explicit target languages, voice design, freeform instructions, emotion/dialect/attribute control, and Tiny multi-speaker dialogue.

## Images and requirements

This project intentionally uses two independently built image tags, overriding the spec’s single-image convention. Each image contains one dependency stack and one virtual environment. A shared multi-stage Dockerfile and CI matrix select the variant.

| Tag | Model | PyTorch / torchaudio | CUDA wheels | transformers |
|---|---|---|---|---|
| `0.5b` | `inclusionAI/Ming-3.0-tts-0.5b` | 2.6.0 | cu124 | 4.46.3 |
| `tiny` | `inclusionAI/Ming-3.0-tts-tiny` | 2.7.1 | cu118 | 4.46.3 |

Both use Python 3.10, an NVIDIA CUDA GPU, and bfloat16; there is no CPU fallback. Upstream reference hardware is H800/H20 for 0.5B and H800 for Tiny. Tiny is a 7.9B-total / 1.3B-active MoE; budget memory for all weights. Each image preserves its upstream dependency versions and omits optional flash-attn.

Runtime dispatch reads the immutable image marker `/opt/api/image-variant`; `MING3_VARIANT` is a build argument only, not an environment setting. Change the image tag to switch model families. `MING3_MODEL` defaults to the matching model and can point to another repository or local checkpoint of the same family.

## Quick start (Docker)

The GHCR names below are intended CI destinations, not a claim that images were built or published during implementation. Replace `openttsgroup` if hosted elsewhere; CI derives the image name from the repository.

```bash
docker pull ghcr.io/openttsgroup/ming3-tts-open-tts:0.5b
# Or: docker pull ghcr.io/openttsgroup/ming3-tts-open-tts:tiny
mkdir -p voices cache
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/ming3-tts-open-tts:0.5b
```
Change the final tag to `:tiny` to deploy Tiny. See [docker/docker-compose.example.yml](./docker/docker-compose.example.yml). First startup downloads weights through the HF SDK. During loading, `/healthz` reports `loading` and synthesis returns 503; loading failure logs the exception and exits. Mount the cache to avoid repeated downloads. Offline deployment uses `HF_HUB_OFFLINE=1` with a populated cache, or a separately mounted checkpoint and `MING3_MODEL`. Image builds never download model weights.

Tiny retains `gcc`, `libc6-dev`, and `python3.10-dev` in the runtime image because FLA/Triton compiles host-side CUDA driver helpers and launchers on demand. This applies to both BF16 and quantized inference. An older image failing with `Failed to find C compiler` must be rebuilt with the updated Dockerfile; setting `CC` alone does not install the missing compiler or headers.

## Experimental Tiny weight quantization

Set `MING3_QUANTIZATION=int8` or `int4` on the `:tiny` image; default is `none`. This is an adapter extension, not an upstream-validated quantization scheme. It is not supported by the `:0.5b` image.

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e MING3_QUANTIZATION=int4 \
  -v "$PWD/voices:/voices:ro" -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/ming3-tts-open-tts:tiny
```

- Convert routed/shared expert MLP projections in the 23 MoE blocks after upstream loads weights on CPU, **before** it moves the model to CUDA. The full BF16 model is never first copied to the GPU on this path.
- Symmetric groupwise weight quantization uses groups of 64. INT4 packs two values per byte; INT8 stores one byte per value, plus group scales. INT4 here is not NF4.
- Routers, embeddings, attention, audio encoder/decoder, DiT and FP32 numeric state retain their original precision. `MING3_DTYPE=bfloat16` remains the floating-point compute dtype; `/healthz.quantization` separately reports expert storage mode.
- Each active expert projection is dequantized on demand and evaluated with ordinary floating-point linear operations. Dequantized weights are not cached. This saves storage; it is **not an optimized low-bit compute kernel**. It can be slower, and INT4 especially needs audio-quality evaluation.
- No bitsandbytes/torchao dependency, upstream edits, or checkpoint rewrites. Unlike Fish's on-disk quantization cache, this implementation reconverts on CPU at every startup. Startup is slower and host RAM must still accommodate the full floating-point model and loading overhead.

For a 12GB RTX 4070 Ti, INT4 offers more storage headroom than INT8, but **neither loading nor synthesis within 12GB has been verified or guaranteed**. Unquantized modules, activations, CUDA workspaces and temporary dequantized tensors also consume VRAM. Logged expert-storage savings are not total peak VRAM; validate with representative text, reference audio and tasks.

CPU unit cases require an existing torch environment but no model weights or GPU: `python -m unittest discover -s tests`. They cover signed packing, zero weights, nonaligned dimensions, error bounds, module selection and loader restoration. Per the request not to run the project, only static syntax checks were performed; these tests and quantized inference were not executed.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MING3_MODEL` | `inclusionAI/Ming-3.0-tts-{variant}` | HF repo ID or mounted local checkpoint; independent of variant. |
| `MING3_REVISION` | `null` | Optional HF model revision; ignored for local paths. |
| `MING3_DEVICE` | `auto` | auto/cuda; cpu is accepted by settings but fails loading explicitly. No CPU support is claimed. |
| `MING3_CUDA_INDEX` | `0` | Visible GPU index, 0..255. |
| `MING3_DTYPE` | `bfloat16` | Only bfloat16 is supported by the public upstream runtime. |
| `MING3_QUANTIZATION` | `none` | Experimental Tiny MoE weight storage: none / int8 / int4; CUDA and :tiny only, other combinations fail. |
| `MING3_TEXT_NORMALIZATION` | `false` | Enable upstream FST normalization; images include pynini. |
| `MING3_DEFAULT_LANGUAGE` | `Chinese` | Default ordinary clone language; choose from /languages. |
| `MING3_DEFAULT_VOICE` | `灵小柔` | Built-in fallback for empty design instructions. |
| `MING3_SEED` | `1234` | Default generation seed for all routes, including dialogue. |
| `MING3_MAX_DECODE_STEPS` | `1000` | 1..16384 per fragment; dialogue uses upstream automatic frame bounds instead. |
| `MING3_MAX_LENGTH` | `24` | Text split target, 1..512; dialogue remains one script. |
| `VOICES_DIR` | `/voices` | Read-only voice catalog. |
| `HOST` | `0.0.0.0` | Listen address. |
| `PORT` | `8000` | HTTP port. |
| `LOG_LEVEL` | `info` | Application and uvicorn log level. |
| `MAX_INPUT_CHARS` | `8000` | Maximum synthesis input length. |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | Default output format. |
| `MAX_AUDIO_BYTES` | `20971520` | Upload limit, 20 MiB. |
| `MAX_CONCURRENCY` | `1` | Fixed to 1: shared runtime state and random generators. Scale with separate containers. |
| `MAX_QUEUE_SIZE` | `0` | Maximum waiters; 0 is unlimited. |
| `QUEUE_TIMEOUT` | `0` | Queue wait seconds; 0 disables timeout. |
| `CORS_ENABLED` | `false` | true enables all origins/methods/headers. |
| `HF_HOME` | `/root/.cache/huggingface` | Persistent SDK cache. |
| `HF_HUB_OFFLINE` | `unset` | Set 1 for cache-only loading. |
| `HF_TOKEN` | `unset` | Optional token for private model repositories. |

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | Readiness, capabilities and queue counters |
| GET | `/v1/audio/voices` | Built-in and file voice catalog |
| GET | `/v1/audio/voices/preview?id=<id>` | File WAV preview; plain ID without file:// |
| GET | `/v1/audio/languages` | Native language keys |
| POST | `/v1/audio/speech` | Built-in or file voice synthesis |
| POST | `/v1/audio/clone` | Multipart upload cloning |
| POST | `/v1/audio/design` | Voice design or freeform generation |
| POST | `/v1/audio/dialogue` | Registered only for Tiny; multi-speaker dialogue |

`/healthz.capabilities`: `clone=true`, `streaming=false`, `design=true`, `languages=true`, `builtin_voices=true`. The upstream API returns complete waveforms, so `/v1/audio/realtime` is not registered. Tiny dialogue is a private extension discoverable from the deployed tag and `/openapi.json`.

Audio is mono at the native rate reported by `/healthz.sample_rate`. PCM is headerless signed int16 little-endian. MIME types: MP3 `audio/mpeg`, Opus `audio/ogg`, AAC `audio/aac`, FLAC `audio/flac`, WAV `audio/wav`, PCM `application/octet-stream`.

Inputs and control combinations are validated before queuing. Unknown voices return 404, size limits 413, invalid upload extensions 415, invalid parameters 422, unsupported remote voice URIs 501, and loading/full queue/queue timeout 503. Inference or audio decoding/encoding failures return 500 and log exceptions. Unknown JSON fields are ignored. Omitted/null `response_format` uses the service default.

### POST /v1/audio/speech

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `input` | string | — | required | Nonblank synthesis text, up to MAX_INPUT_CHARS. Inline paralinguistic tags are preserved. |
| `voice` | string | — | required | Built-in ID from /voices, or file://<id>. |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm. Omitted/null uses DEFAULT_RESPONSE_FORMAT. |
| `model` | string / null | null | ignored | OpenAI compatibility only; never changes the loaded checkpoint. |
| `speed` | number | 1.0 | ignored | Accepted range 0.25..4.0; use attribute_speed for native categorical speed. |
| `instructions` | string / null | null | ignored | Accepted without error. Use /design with mode=freeform for instruction-only generation. |
| `language` | string / null | Chinese | extension | Chinese / English / German / Spanish / French / Korean / Japanese. Only ordinary cloning consumes it; ignored for built-in and controlled tasks. Default follows MING3_DEFAULT_LANGUAGE. |
| `emotion` | string / null | null | extension | 高兴 / 悲伤 / 愤怒 / 恐惧 / 厌恶 / 惊讶 / 中性; speaker-only EMOTION task. |
| `dialect` | string / null | null | extension | cantonese / chuanyu dialect; speaker-only DIALECT task. |
| `attribute_speed` | string / null | null | extension | 慢速 / 中速 / 快速; activates BASIC. Omitted BASIC speed defaults to 中速. |
| `pitch` | string / null | null | extension | 低 / 中 / 高; activates BASIC. Omitted BASIC pitch defaults to 中. |
| `volume` | string / null | null | extension | 低 / 中 / 高; activates BASIC. Omitted BASIC volume defaults to 中. |
| `seed` | integer / null | 1234 | extension | Generation seed, 0..4294967295. |
| `cfg_scale` | number / null | 2.0 | extension | Guidance scale, 0..10. |
| `temperature` | number / null | 0.0 | extension | Sampling temperature, 0..2. |
| `sigma` | number / null | 0.25 | extension | Flow noise scale, 0..1. |
A `file://` voice selects cloning; a plain ID selects the IP task. Emotion, dialect and BASIC are mutually exclusive; the three BASIC attributes may be combined. Combining those controls with built-in voices returns 422. Ordinary cloning uses reference audio and transcript; controlled tasks use speaker conditioning without reference text or target language. Inline `[Sigh]`, `[Laughter]` and other tags can be used for ordinary Chinese/English cloning.

### POST /v1/audio/clone

`multipart/form-data`

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `audio` | file | — | required | Reference audio; maximum 20 MiB by default. |
| `prompt_text` | string | — | required | Transcript required for ordinary cloning. For emotion/dialect/BASIC, the field is accepted but its content is ignored. |
| `input` | string | — | required | Nonblank synthesis text, up to MAX_INPUT_CHARS. Inline paralinguistic tags are preserved. |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm. Omitted/null uses DEFAULT_RESPONSE_FORMAT. |
| `model` | string / null | null | ignored | OpenAI compatibility only; never changes the loaded checkpoint. |
| `speed` | number | 1.0 | ignored | Accepted range 0.25..4.0; use attribute_speed for native categorical speed. |
| `instructions` | string / null | null | ignored | Accepted without error. Use /design with mode=freeform for instruction-only generation. |
| `language` | string / null | Chinese | extension | Chinese / English / German / Spanish / French / Korean / Japanese. Only ordinary cloning consumes it; ignored for built-in and controlled tasks. Default follows MING3_DEFAULT_LANGUAGE. |
| `emotion` | string / null | null | extension | 高兴 / 悲伤 / 愤怒 / 恐惧 / 厌恶 / 惊讶 / 中性; speaker-only EMOTION task. |
| `dialect` | string / null | null | extension | cantonese / chuanyu dialect; speaker-only DIALECT task. |
| `attribute_speed` | string / null | null | extension | 慢速 / 中速 / 快速; activates BASIC. Omitted BASIC speed defaults to 中速. |
| `pitch` | string / null | null | extension | 低 / 中 / 高; activates BASIC. Omitted BASIC pitch defaults to 中. |
| `volume` | string / null | null | extension | 低 / 中 / 高; activates BASIC. Omitted BASIC volume defaults to 中. |
| `seed` | integer / null | 1234 | extension | Generation seed, 0..4294967295. |
| `cfg_scale` | number / null | 2.0 | extension | Guidance scale, 0..10. |
| `temperature` | number / null | 0.0 | extension | Sampling temperature, 0..2. |
| `sigma` | number / null | 0.25 | extension | Flow noise scale, 0..1. |
Accepts WAV/MP3/FLAC/OGG/Opus/M4A/AAC/WebM. PyAV decodes references into temporary mono WAV files; upstream resamples for the model. Uploads are never stored in the voice catalog and are cleaned after inference or failure. Prompt caching is disabled, so changed catalog files and temporary uploads cannot reuse stale prompts.

### POST /v1/audio/design

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `input` | string | — | required | Nonblank synthesis text, up to MAX_INPUT_CHARS. Inline paralinguistic tags are preserved. |
| `instruct` | string / null | null | supported | Voice description, or freeform instruction. Blank/null selects MING3_DEFAULT_VOICE. |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm. Omitted/null uses DEFAULT_RESPONSE_FORMAT. |
| `mode` | string / null | voice_design | extension | voice_design / freeform. No reference audio is used. |
| `seed` | integer / null | 1234 | extension | Generation seed, 0..4294967295. |
| `cfg_scale` | number / null | 2.0 | extension | Guidance scale, 0..10. |
| `temperature` | number / null | 0.0 | extension | Sampling temperature, 0..2. |
| `sigma` | number / null | 0.25 | extension | Flow noise scale, 0..1. |
`voice_design` maps to the upstream style-description task; `freeform` maps to instruction-only generation, both without references. Blank descriptions select the default built-in voice. This endpoint does not consume OpenAI `model`, `speed` or `instructions` fields.

### POST /v1/audio/dialogue

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `input` | string | — | required | One speaker_N: utterance per line. All IDs 1..N must appear. |
| `voices` | array[string] | — | required | 2..16 file:// IDs, ordered by speaker number. Built-in voices are not accepted. |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm. Omitted/null uses DEFAULT_RESPONSE_FORMAT. |
| `seed` | integer / null | 1234 | extension | Generation seed, 0..4294967295. |
| `cfg_scale` | number / null | 2.0 | extension | Guidance scale, 0..10. |
| `temperature` | number / null | 0.0 | extension | Sampling temperature, 0..2. |
| `sigma` | number / null | 0.25 | extension | Flow noise scale, 0..1. |
Only the `:tiny` image registers this route; 0.5B returns 404. Language is inferred from the script, and upstream computes generation-frame bounds. The adapter uses seed=1234 consistently; pass `seed=1235` for the upstream podcast seed default.

## Examples

```bash
# Built-in voice
curl http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello, welcome to Ming.","voice":"US_Emma","response_format":"wav"}' -o speech.wav

# File voice, English cloning
curl http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello again.","voice":"file://alice","language":"English"}' -o clone.mp3

# Temporary reference upload
curl http://localhost:8000/v1/audio/clone \
  -F audio=@reference.wav -F 'prompt_text=This is my reference recording.' \
  -F 'input=This is a new sentence.' -F language=English -o upload.mp3

# Text-only voice design
curl http://localhost:8000/v1/audio/design -H 'Content-Type: application/json' \
  -d '{"input":"Good evening.","instruct":"A warm, calm narrator."}' -o design.mp3

# Freeform delivery
curl http://localhost:8000/v1/audio/design -H 'Content-Type: application/json' \
  -d '{"input":"The story begins here.","instruct":"Speak softly, with a hint of nostalgia.","mode":"freeform"}' -o freeform.mp3

# Tiny only; alice and bob must exist in /voices
curl http://localhost:8000/v1/audio/dialogue -H 'Content-Type: application/json' \
  -d '{"input":"speaker_1: Hello!\nspeaker_2: Welcome back.","voices":["file://alice","file://bob"],"seed":1235}' -o dialogue.mp3
```

## Voice directory

```text
voices/
  alice.wav
  alice.txt
  alice.yml  # optional
  bob.wav
  bob.txt
```

Only top-level files are scanned. WAV and UTF-8 transcript must both exist and be nonempty; BOM is supported. Optional YAML must be a mapping; malformed metadata does not disable the voice. Voice IDs use `file://alice`; preview queries use `alice`. Symlinks escaping the catalog are excluded. Built-in preview_url, prompt_text and metadata are null. `app/builtin_voices.json` snapshots 241 callable IDs from the pinned upstream `cookbooks/instructions.md` tables; review it when updating upstream.

## Building, development and publishing

These are instructions for subsequent use; they were not executed during implementation.

```bash
git submodule update --init --recursive
# Dense 0.5B
docker build -f docker/Dockerfile -t ming3-tts-open-tts:0.5b .
# Tiny, isolated from the dense dependencies
docker build -f docker/Dockerfile -t ming3-tts-open-tts:tiny \
  --build-arg MING3_VARIANT=tiny \
  --build-arg CUDA_VERSION=11.8.0 --build-arg CUDNN_VARIANT=cudnn8 \
  --build-arg TORCH_VERSION=2.7.1 --build-arg TORCH_CUDA_TAG=cu118 .
```
For local development, use a separate Python 3.10 environment per model. Install matching torch/torchaudio from the corresponding CUDA index first, then the selected `engine/requirements*.txt`, `docker/requirements.api.txt` and `docker/requirements.engine.txt`. For source-only development, create an `image-variant` file at the repository root containing `0.5b` or `tiny` to match the installed dependencies (this local file is ignored by Git and Docker). Set `MING3_MODEL` if overriding the default, and `VOICES_DIR`, then run `uvicorn app.server:app --host 0.0.0.0 --port 8000` from the repository root, with one worker. The adapter adds `engine` to its import path.

CI independently builds both variants on main pushes, v* tags and manual dispatch. Branch tags are `0.5b` / `tiny`; commit tags include `0.5b-sha-abcdef0`; release tags include `1.2.3-0.5b` / `1.2.3-tiny`. No ambiguous `latest` tag is published. Build caches use separate scopes.

## Validation scope

Implementation was checked statically using Python 3.10 AST parsing, layout/API inspection, upstream signatures, and configuration consistency. No dependencies were installed, images built, service started, model weights downloaded or inference executed. Audio quality, dependency resolution, CUDA and end-to-end behavior still require deployment validation.
