# syntax=docker/dockerfile:1.7
# One variant per image; CI publishes :0.5b and :tiny. No model weights.
ARG MING3_VARIANT=0.5b
ARG CUDA_VERSION=12.4.1
ARG CUDNN_VARIANT=cudnn
ARG UBUNTU_VERSION=22.04
ARG TORCH_CUDA_TAG=cu124
ARG TORCH_VERSION=2.6.0

FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_VARIANT}-runtime-ubuntu${UBUNTU_VERSION} AS builder
ARG TORCH_CUDA_TAG
ARG TORCH_VERSION
ARG MING3_VARIANT
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}

COPY engine /build/engine
# Preserve the selected upstream dependency set, except torch (installed above) and its
# wheel-index directive. Constraints also prevent later transitive upgrades.
RUN case "$MING3_VARIANT" in \
        0.5b) requirements=/build/engine/requirements.txt ;; \
        tiny) requirements=/build/engine/requirements-tiny.txt ;; \
        *) echo "Invalid MING3_VARIANT" >&2; exit 2 ;; \
    esac \
    && pip freeze > /tmp/constraints.txt \
    && sed -E '/^(torch|torchaudio)==/d; /^--extra-index-url/d' "$requirements" > /tmp/engine.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/engine.txt
COPY docker/requirements.api.txt docker/requirements.engine.txt /tmp/
RUN pip freeze > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/requirements.api.txt -r /tmp/requirements.engine.txt \
    && pip check
RUN find /opt/venv -type d -name __pycache__ -exec rm -rf {} +

FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_VARIANT}-runtime-ubuntu${UBUNTU_VERSION}
ARG MING3_VARIANT
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    HF_HOME=/root/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/opt/api/engine PATH=/opt/venv/bin:$PATH \
    MING3_MODEL=inclusionAI/Ming-3.0-tts-${MING3_VARIANT}
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 ffmpeg libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
WORKDIR /opt/api
COPY engine /opt/api/engine
COPY app /opt/api/app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && printf '%s\n' "$MING3_VARIANT" > /opt/api/image-variant
VOLUME ["/voices", "/root/.cache"]
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD []
