# ming3-tts-open-tts

[English](./README.md) · **中文**

基于 [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) 的 [Open TTS](https://github.com/OpenTTSGroup/open-tts-spec) 适配。支持 0.5B 与 Tiny 的文件／上传克隆、241 个内置音色、七种显式目标语言、声音设计、自由指令、情绪／方言／属性控制，以及 Tiny 多说话人对话。

## 镜像与运行要求

按本项目要求，采用两个独立镜像标签，覆盖规范的单镜像约定。每个镜像只安装一套依赖、一个虚拟环境；使用同一个多阶段 Dockerfile，通过构建参数和 CI 矩阵选择版本。

| Tag | Model | PyTorch / torchaudio | CUDA wheels | transformers |
|---|---|---|---|---|
| `0.5b` | `inclusionAI/Ming-3.0-tts-0.5b` | 2.6.0 | cu124 | 4.46.3 |
| `tiny` | `inclusionAI/Ming-3.0-tts-tiny` | 2.7.1 | cu118 | 4.46.3 |

均使用 Python 3.10、NVIDIA CUDA GPU 和 bfloat16；不提供 CPU 回退。上游参考硬件为 0.5B 的 H800/H20，以及 Tiny 的 H800。Tiny 是总参数 7.9B、激活 1.3B 的 MoE，显存按完整权重考虑。两个镜像保留各自的上游依赖版本，均不安装可选 flash-attn。

运行时直接读取镜像内的固定标记 `/opt/api/image-variant` 选择接口；`MING3_VARIANT` 仅保留为构建参数，不再是环境变量配置。切换模型族时更换镜像 tag。`MING3_MODEL` 默认匹配镜像版本，也可指向同族的其他模型仓库或本地权重。

## Docker 快速开始

以下 GHCR 地址是 CI 的预期发布位置，并不表示本次已经构建或发布。若仓库属于其他组织，替换 `openttsgroup`；CI 根据实际仓库名自动计算地址。

```bash
docker pull ghcr.io/openttsgroup/ming3-tts-open-tts:0.5b
# Or: docker pull ghcr.io/openttsgroup/ming3-tts-open-tts:tiny
mkdir -p voices cache
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/ming3-tts-open-tts:0.5b
```
将最后一行 tag 改为 `:tiny` 即部署 Tiny。Compose 示例见 [docker/docker-compose.example.yml](./docker/docker-compose.example.yml)。首次启动由 HF SDK 自动下载权重；下载期间 `/healthz` 返回 `loading`，合成返回 503；失败会记录异常并退出。挂载缓存避免重复下载。离线部署用 `HF_HUB_OFFLINE=1` 与已有缓存，或额外挂载本地模型目录并设置 `MING3_MODEL`。构建镜像不下载权重。

## Tiny 实验性权重量化

`:tiny` 镜像可设置 `MING3_QUANTIZATION=int8` 或 `int4`，默认 `none`。这是适配层新增的功能，不是上游官方验证的量化方案；不适用于 `:0.5b` 镜像。

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e MING3_QUANTIZATION=int4 \
  -v "$PWD/voices:/voices:ro" -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/ming3-tts-open-tts:tiny
```

- 在上游 CPU 权重加载完成、整模型转入 CUDA **之前**转换 23 个 MoE 层的路由专家及共享专家 MLP 投影，避免先把完整 BF16 模型塞进 GPU。
- 使用分组大小 64 的对称权重量化；INT4 每字节打包两个值，INT8 每值一个字节，另存分组 scale。这里的 INT4 不是 NF4。
- 路由器、嵌入、注意力、音频编码器／解码器、DiT 和 FP32 数值状态保留原精度。`MING3_DTYPE=bfloat16` 仍表示浮点计算精度；`/healthz.quantization` 单独报告专家权重存储模式。
- 每次只解量化正在执行的一个专家投影，随后使用普通浮点 linear 计算，不缓存全部解量化权重。这是**节省显存的实现，不是低比特加速内核**，可能更慢，INT4 尤其需要评估音质。
- 不增加 bitsandbytes/torchao 等依赖，不修改上游子模块，不写回原始 checkpoint。与 Fish 的磁盘量化缓存不同，本实现每次启动在 CPU 上重新转换，因此启动更慢，仍需足够主机内存容纳完整浮点模型和加载开销。

对 12GB RTX 4070 Ti，INT4 是比 INT8 更有余量的尝试方向，但**尚未实测，不保证可在 12GB 内完成加载和合成**。未量化模块、激活、CUDA 工作区和临时解量化张量仍占显存。启动日志报告的专家存储压缩量不是整机峰值显存；需用实际文本、参考音频和目标任务验证。

提供了无需模型权重或 GPU 的 CPU 单元测试（需要已有 torch 环境）：`python -m unittest discover -s tests`，覆盖打包符号、零权重、非整组维度、误差边界、模块筛选与加载钩子恢复。本次遵照不运行项目的要求，仅做静态语法检查，未执行这些测试或量化推理。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MING3_MODEL` | `inclusionAI/Ming-3.0-tts-{variant}` | HF 仓库 ID 或已挂载的本地模型路径，与 variant 独立。 |
| `MING3_REVISION` | `null` | 可选 HF 模型版本，本地路径忽略。 |
| `MING3_DEVICE` | `auto` | auto/cuda；配置接受 cpu，但加载时明确报错，不宣称支持 CPU。 |
| `MING3_CUDA_INDEX` | `0` | 容器可见 GPU 编号，0..255。 |
| `MING3_DTYPE` | `bfloat16` | 上游公开运行接口仅支持 bfloat16。 |
| `MING3_QUANTIZATION` | `none` | 实验性 Tiny MoE 专家权重量化：none / int8 / int4；仅 CUDA、仅 :tiny，其他组合报错。 |
| `MING3_TEXT_NORMALIZATION` | `false` | 启用上游 FST 文本归一化；镜像包含 pynini。 |
| `MING3_DEFAULT_LANGUAGE` | `Chinese` | 普通克隆默认语言，取值见 /languages。 |
| `MING3_DEFAULT_VOICE` | `灵小柔` | 声音设计描述为空时使用的内置音色。 |
| `MING3_SEED` | `1234` | 所有端点默认随机种子，包括对话。 |
| `MING3_MAX_DECODE_STEPS` | `1000` | 每片段 1..16384；对话改用上游自动计算的帧边界。 |
| `MING3_MAX_LENGTH` | `24` | 文本切分目标长度，1..512；对话脚本保持整体。 |
| `VOICES_DIR` | `/voices` | 只读声音目录。 |
| `HOST` | `0.0.0.0` | 监听地址。 |
| `PORT` | `8000` | HTTP 端口。 |
| `LOG_LEVEL` | `info` | 应用及 uvicorn 日志级别。 |
| `MAX_INPUT_CHARS` | `8000` | 合成文本长度上限。 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | 默认音频输出格式。 |
| `MAX_AUDIO_BYTES` | `20971520` | 上传大小上限，20 MiB。 |
| `MAX_CONCURRENCY` | `1` | 固定为 1，保护共享运行状态和随机数生成器；可用多个容器扩容。 |
| `MAX_QUEUE_SIZE` | `0` | 最大排队请求数，0 不限制。 |
| `QUEUE_TIMEOUT` | `0` | 排队超时秒数，0 不超时。 |
| `CORS_ENABLED` | `false` | true 放开全部来源、方法和请求头。 |
| `HF_HOME` | `/root/.cache/huggingface` | 持久化 SDK 缓存。 |
| `HF_HUB_OFFLINE` | `unset` | 设为 1 时仅使用本地缓存。 |
| `HF_TOKEN` | `unset` | 访问私有模型仓库的可选令牌。 |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 模型状态、能力和并发统计 |
| GET | `/v1/audio/voices` | 内置音色和文件声音列表 |
| GET | `/v1/audio/voices/preview?id=<id>` | 文件声音 WAV 预览，id 不带 file:// |
| GET | `/v1/audio/languages` | 原生语言标识列表 |
| POST | `/v1/audio/speech` | 内置音色或文件声音合成 |
| POST | `/v1/audio/clone` | multipart 上传克隆 |
| POST | `/v1/audio/design` | 声音设计或自由指令 |
| POST | `/v1/audio/dialogue` | 仅 Tiny 注册，多说话人对话 |

`/healthz.capabilities`：`clone=true`、`streaming=false`、`design=true`、`languages=true`、`builtin_voices=true`。上游接口返回完整波形，不暴露 `/v1/audio/realtime`。Tiny 对话是私有扩展，以部署 tag 和 `/openapi.json` 发现。

响应为单声道音频，原生采样率由 `/healthz.sample_rate` 提供。PCM 为无头 signed int16 小端。MIME：MP3 `audio/mpeg`、Opus `audio/ogg`、AAC `audio/aac`、FLAC `audio/flac`、WAV `audio/wav`、PCM `application/octet-stream`。

输入及控制组合在排队前校验。未知声音 404，超限 413，非法格式扩展名 415，参数错误 422，不支持的远程声音 URI 501，加载／队列满／排队超时 503。推理或音频解码／编码失败为 500，并记录异常。未知 JSON 字段忽略；`response_format` 省略或 null 时使用服务默认值。

### POST /v1/audio/speech

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `input` | string | — | required | 非空合成文本，不超过 MAX_INPUT_CHARS；保留行内副语言标签。 |
| `voice` | string | — | required | /voices 返回的内置音色 ID 或 file://<id>。 |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm。省略或 null 使用 DEFAULT_RESPONSE_FORMAT。 |
| `model` | string / null | null | ignored | 仅用于 OpenAI 兼容，不切换模型。 |
| `speed` | number | 1.0 | ignored | 接受 0.25..4.0，但不生效；原生分档语速使用 attribute_speed。 |
| `instructions` | string / null | null | ignored | 接受但忽略；自由指令生成请调用 /design 并设置 mode=freeform。 |
| `language` | string / null | Chinese | extension | Chinese / English / German / Spanish / French / Korean / Japanese。仅普通克隆使用；内置音色和控制任务忽略。默认由 MING3_DEFAULT_LANGUAGE 决定。 |
| `emotion` | string / null | null | extension | 高兴 / 悲伤 / 愤怒 / 恐惧 / 厌恶 / 惊讶 / 中性；使用仅说话人嵌入的 EMOTION 任务。 |
| `dialect` | string / null | null | extension | cantonese / chuanyu dialect，分别为粤语、川渝话；使用 DIALECT 任务。 |
| `attribute_speed` | string / null | null | extension | 慢速 / 中速 / 快速，启用 BASIC 任务；该任务未给此项时用中速。 |
| `pitch` | string / null | null | extension | 低 / 中 / 高，启用 BASIC 任务；该任务未给此项时用中。 |
| `volume` | string / null | null | extension | 低 / 中 / 高，启用 BASIC 任务；该任务未给此项时用中。 |
| `seed` | integer / null | 1234 | extension | 随机种子，0..4294967295。 |
| `cfg_scale` | number / null | 2.0 | extension | 引导强度，0..10。 |
| `temperature` | number / null | 0.0 | extension | 采样温度，0..2。 |
| `sigma` | number / null | 0.25 | extension | 流匹配噪声系数，0..1。 |
以 `file://` 开头的声音走克隆；普通 ID 走 IP 内置音色。emotion、dialect、BASIC 三类控制互斥，BASIC 可同时设置三个属性；内置音色与这些控制组合返回 422。普通克隆携带参考音频和文本；控制任务只使用说话人信息，不传参考文本或目标语言。行内 `[Sigh]`、`[Laughter]` 等标签可用于普通中英文克隆。

### POST /v1/audio/clone

`multipart/form-data`

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `audio` | file | — | required | 参考音频，默认上限 20 MiB。 |
| `prompt_text` | string | — | required | 普通克隆必须提供非空参考文本；情绪／方言／BASIC 任务接受但忽略文本内容。 |
| `input` | string | — | required | 非空合成文本，不超过 MAX_INPUT_CHARS；保留行内副语言标签。 |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm。省略或 null 使用 DEFAULT_RESPONSE_FORMAT。 |
| `model` | string / null | null | ignored | 仅用于 OpenAI 兼容，不切换模型。 |
| `speed` | number | 1.0 | ignored | 接受 0.25..4.0，但不生效；原生分档语速使用 attribute_speed。 |
| `instructions` | string / null | null | ignored | 接受但忽略；自由指令生成请调用 /design 并设置 mode=freeform。 |
| `language` | string / null | Chinese | extension | Chinese / English / German / Spanish / French / Korean / Japanese。仅普通克隆使用；内置音色和控制任务忽略。默认由 MING3_DEFAULT_LANGUAGE 决定。 |
| `emotion` | string / null | null | extension | 高兴 / 悲伤 / 愤怒 / 恐惧 / 厌恶 / 惊讶 / 中性；使用仅说话人嵌入的 EMOTION 任务。 |
| `dialect` | string / null | null | extension | cantonese / chuanyu dialect，分别为粤语、川渝话；使用 DIALECT 任务。 |
| `attribute_speed` | string / null | null | extension | 慢速 / 中速 / 快速，启用 BASIC 任务；该任务未给此项时用中速。 |
| `pitch` | string / null | null | extension | 低 / 中 / 高，启用 BASIC 任务；该任务未给此项时用中。 |
| `volume` | string / null | null | extension | 低 / 中 / 高，启用 BASIC 任务；该任务未给此项时用中。 |
| `seed` | integer / null | 1234 | extension | 随机种子，0..4294967295。 |
| `cfg_scale` | number / null | 2.0 | extension | 引导强度，0..10。 |
| `temperature` | number / null | 0.0 | extension | 采样温度，0..2。 |
| `sigma` | number / null | 0.25 | extension | 流匹配噪声系数，0..1。 |
支持 WAV/MP3/FLAC/OGG/Opus/M4A/AAC/WebM。适配层使用 PyAV 解码并转为临时单声道 WAV，上游继续按模型需要重采样。上传不保存到声音目录；推理结束或失败后清理。提示缓存关闭，目录文件更新和临时上传不会命中过期缓存。

### POST /v1/audio/design

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `input` | string | — | required | 非空合成文本，不超过 MAX_INPUT_CHARS；保留行内副语言标签。 |
| `instruct` | string / null | null | supported | 音色描述或自由指令。空白或 null 使用 MING3_DEFAULT_VOICE。 |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm。省略或 null 使用 DEFAULT_RESPONSE_FORMAT。 |
| `mode` | string / null | voice_design | extension | voice_design / freeform；均不使用参考音频。 |
| `seed` | integer / null | 1234 | extension | 随机种子，0..4294967295。 |
| `cfg_scale` | number / null | 2.0 | extension | 引导强度，0..10。 |
| `temperature` | number / null | 0.0 | extension | 采样温度，0..2。 |
| `sigma` | number / null | 0.25 | extension | 流匹配噪声系数，0..1。 |
`voice_design` 映射上游风格描述任务；`freeform` 映射自由指令任务，均无参考音频。空描述直接使用默认内置音色。此端点不使用 OpenAI 的 `model`、`speed` 或 `instructions` 字段。

### POST /v1/audio/dialogue

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `input` | string | — | required | 每行 speaker_N: 文本，必须覆盖从 1 到 N 的所有说话人。 |
| `voices` | array[string] | — | required | 2..16 个 file:// ID，按说话人编号排序；不接受内置音色。 |
| `response_format` | string / null | mp3 | supported | mp3 / opus / aac / flac / wav / pcm。省略或 null 使用 DEFAULT_RESPONSE_FORMAT。 |
| `seed` | integer / null | 1234 | extension | 随机种子，0..4294967295。 |
| `cfg_scale` | number / null | 2.0 | extension | 引导强度，0..10。 |
| `temperature` | number / null | 0.0 | extension | 采样温度，0..2。 |
| `sigma` | number / null | 0.25 | extension | 流匹配噪声系数，0..1。 |
仅 `:tiny` 镜像注册，0.5B 返回 404。语言由脚本推断，生成帧边界由上游计算。适配器统一默认 seed=1234；复现上游播客默认值时显式传 `seed=1235`。

## 请求示例

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

## 声音目录

```text
voices/
  alice.wav
  alice.txt
  alice.yml  # optional
  bob.wav
  bob.txt
```

仅扫描顶层，WAV 与 UTF-8 文本必须同时存在且非空；支持 BOM。YAML 元信息可选，必须为对象，解析错误不影响音色。声音 ID 为 `file://alice`，预览参数为 `alice`。指向目录外的符号链接被排除。内置音色的 preview_url、prompt_text、metadata 均为 null。`app/builtin_voices.json` 是固定上游版本 `cookbooks/instructions.md` 内置音色表的 241 个调用 ID 快照；更新上游时同步检查。

## 构建、开发与发布

以下仅为后续操作说明，本次未执行这些命令。

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
本地开发同样应为每个模型使用独立 Python 3.10 环境，按对应 CUDA 索引先安装匹配的 torch/torchaudio，再安装对应 `engine/requirements*.txt` 和 `docker/requirements.api.txt`、`docker/requirements.engine.txt`。源码开发时，在项目根创建 `image-variant` 文件，内容为与已安装依赖匹配的 `0.5b` 或 `tiny`（该本地文件被 Git 与 Docker 忽略）。按需覆盖 `MING3_MODEL` 并设置 `VOICES_DIR`，从项目根以 `uvicorn app.server:app --host 0.0.0.0 --port 8000` 启动；只启动一个 worker。应用自行将 `engine` 加入导入路径。

CI 在 main 推送、v* 标签、手动触发时分别构建两套镜像。分支标签为 `0.5b` / `tiny`，提交标签如 `0.5b-sha-abcdef0`，版本标签如 `1.2.3-0.5b` / `1.2.3-tiny`，不发布含义不明的 `latest`。两个构建使用独立缓存 scope。

## 验证范围

本次只做 Python 3.10 AST 语法、目录／接口／上游签名与配置的静态核对。没有安装依赖、构建镜像、启动服务、下载模型或执行推理；音质、依赖解析、CUDA 与端到端行为仍需部署环境验证。
