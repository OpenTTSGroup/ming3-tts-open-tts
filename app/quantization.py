from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Literal

import torch
from torch import nn
from torch.nn import functional as F

log = logging.getLogger(__name__)
QuantizationMode = Literal["int8", "int4"]
_LOAD_LOCK = threading.Lock()


class WeightOnlyLinear(nn.Module):
    """Groupwise symmetric storage; dequantize one expert projection per call."""

    def __init__(self, linear: nn.Linear, mode: QuantizationMode) -> None:
        super().__init__()
        if mode not in ("int8", "int4"):
            raise ValueError(f"Unsupported quantization: {mode}")
        if linear.weight.device.type != "cpu":
            raise ValueError("Quantization must precede the model's CUDA transfer")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.mode = mode
        self.group_size = 64
        # Only this projection is expanded to FP32, not the entire model.
        with torch.no_grad():
            weight = linear.weight.detach().float()
            if not torch.isfinite(weight).all():
                raise ValueError("Cannot quantize non-finite expert weights")
            padding = (-self.in_features) % self.group_size
            weight = F.pad(weight, (0, padding)).reshape(self.out_features, -1, self.group_size)
            limit = 127 if mode == "int8" else 7
            scales = (weight.abs().amax(dim=-1, keepdim=True) / limit).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            quantized = (weight / scales).round().clamp(-limit, limit).to(torch.int8)
            if mode == "int4":
                unsigned = (quantized.to(torch.int16) + 8).to(torch.uint8)
                quantized = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
            self.register_buffer("qweight", quantized.contiguous())
            self.register_buffer("scales", scales)
            self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        self.eval()

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        if self.mode == "int4":
            unpacked = torch.stack((self.qweight & 15, self.qweight >> 4), dim=-1)
            weight = unpacked.reshape(self.out_features, -1, self.group_size).to(dtype) - 8
        else:
            weight = self.qweight.to(dtype)
        weight = weight * self.scales.to(dtype)
        return weight.reshape(self.out_features, -1)[:, :self.in_features].contiguous()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Deliberately no dequantized-weight cache: that would grow back toward
        # full BF16 residency as routing visits more experts.
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return F.linear(inputs, self.dequantize(inputs.dtype), bias)


def quantize_experts(model: nn.Module, mode: QuantizationMode) -> dict[str, int]:
    """Quantize only routed/shared MoE MLPs, leaving numeric state untouched."""
    candidates: list[tuple[nn.Module, str]] = []
    routed_blocks = 0
    for name, block in model.named_modules():
        if not name.endswith(".mlp") or not hasattr(block, "experts"):
            continue
        if not isinstance(block.experts, nn.ModuleList):
            raise RuntimeError("Weight-only quantization requires the expanded loop MoE backend")
        routed_blocks += 1
        experts = list(block.experts)
        shared = getattr(block, "shared_experts", None)
        if shared is not None:
            experts.append(shared)
        for expert in experts:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                layer = getattr(expert, projection, None)
                if not isinstance(layer, nn.Linear) or layer.weight.device.type != "cpu":
                    raise RuntimeError(f"Unexpected CPU expert projection: {name}.{projection}")
                candidates.append((expert, projection))
    # Guard the pinned release layout instead of silently quantizing nothing
    # after an incompatible upstream update.
    if routed_blocks != 23:
        raise RuntimeError(f"Expected 23 Tiny MoE blocks; found {routed_blocks}")
    original_bytes = packed_bytes = 0
    for parent, name in candidates:
        layer = getattr(parent, name)
        original_bytes += sum(t.numel() * t.element_size() for t in layer.parameters())
        replacement = WeightOnlyLinear(layer, mode)
        packed_bytes += sum(t.numel() * t.element_size() for t in replacement.buffers())
        setattr(parent, name, replacement)
        del layer
    stats = dict(modules=len(candidates), original_bytes=original_bytes, packed_bytes=packed_bytes)
    log.info("Tiny %s expert storage: %d projections, %.2f -> %.2f GiB (not total VRAM)",
             mode, stats["modules"], original_bytes / 2**30, packed_bytes / 2**30)
    return stats


@contextmanager
def quantized_tiny_loading(mode: QuantizationMode) -> Iterator[None]:
    """Intercept CPU checkpoint loading, before upstream calls model.to(cuda)."""
    from ming_tiny.task_protocol import configure_release_environment

    # Upstream normally establishes these flags before importing model code.
    configure_release_environment()
    from ming_tiny.modeling_lingv3_tts import LingV3TTSForConditionalGeneration

    cls = LingV3TTSForConditionalGeneration
    with _LOAD_LOCK:
        original = cls.from_pretrained
        own_descriptor = cls.__dict__.get("from_pretrained")

        def load_quantized(owner, *args, **kwargs):
            model = original(*args, **kwargs)
            quantize_experts(model, mode)
            return model

        cls.from_pretrained = classmethod(load_quantized)
        try:
            yield
        finally:
            # Restore the descriptor even if loading or conversion fails.
            if own_descriptor is None:
                delattr(cls, "from_pretrained")
            else:
                cls.from_pretrained = own_descriptor
