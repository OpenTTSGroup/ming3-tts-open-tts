"""CPU-only unit cases; no model checkpoints, network, or service startup.

Run in an existing dependency environment: python -m unittest discover -s tests
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from app.quantization import WeightOnlyLinear, quantize_experts, quantized_tiny_loading


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(65, 9, bias=False)
        self.up_proj = nn.Linear(65, 9, bias=False)
        self.down_proj = nn.Linear(9, 65, bias=False)


def tiny_shape(blocks: int = 23) -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList()
    for _ in range(blocks):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.ModuleList([Expert(), Expert()])
        layer.mlp.shared_experts = Expert()
        layer.mlp.gate = nn.Linear(65, 2)
        layer.attention = nn.Linear(65, 65)
        model.layers.append(layer)
    model.audio = nn.Linear(65, 65)
    return model


class QuantizationTests(unittest.TestCase):
    def test_signed_nibbles_zeros_bias_and_odd_width(self) -> None:
        layer = nn.Linear(15, 3)
        with torch.no_grad():
            layer.weight[0] = torch.arange(-7, 8)
            layer.weight[1] = 0
            layer.weight[2] = torch.arange(7, -8, -1)
            layer.bias.copy_(torch.tensor([1., -2., 3.]))
        quantized = WeightOnlyLinear(layer, "int4")
        inputs = torch.arange(30, dtype=torch.float32).reshape(2, 15)
        torch.testing.assert_close(quantized(inputs), layer(inputs), atol=0, rtol=0)
        self.assertEqual(quantized.qweight.dtype, torch.uint8)
        self.assertEqual(quantized.qweight.numel(), 3 * 32)
        self.assertNotIn("weight", dict(quantized.named_parameters()))

    def test_quantization_error_bound_and_arbitrary_batch_shape(self) -> None:
        torch.manual_seed(7)
        for mode, limit in (("int8", 127), ("int4", 7)):
            with self.subTest(mode=mode):
                layer = nn.Linear(67, 11)
                inputs = torch.randn(2, 3, 67)
                quantized = WeightOnlyLinear(layer, mode)
                # Uniform symmetric quantization error is bounded by half a
                # step per weight; dot-product error by the weighted L1 bound.
                bound = inputs.abs().sum(-1, keepdim=True) * layer.weight.abs().max() / (2 * limit)
                error = (quantized(inputs) - layer(inputs)).abs()
                self.assertTrue(torch.all(error <= bound + 1e-5))
                self.assertEqual(tuple(quantized(inputs).shape), (2, 3, 11))
                self.assertEqual(quantized.qweight.dtype, torch.int8 if mode == "int8" else torch.uint8)

    def test_only_experts_are_replaced_and_buffers_stay_packed(self) -> None:
        model = tiny_shape()
        router = model.layers[0].mlp.gate
        attention = model.layers[0].attention
        audio = model.audio
        stats = quantize_experts(model, "int4")
        self.assertEqual(stats["modules"], 23 * 3 * 3)
        self.assertLess(stats["packed_bytes"], stats["original_bytes"])
        self.assertIs(model.layers[0].mlp.gate, router)
        self.assertIs(model.layers[0].attention, attention)
        self.assertIs(model.audio, audio)
        model.to(dtype=torch.bfloat16)
        expert = model.layers[0].mlp.experts[0].gate_proj
        self.assertEqual(expert.qweight.dtype, torch.uint8)
        self.assertEqual(expert.dequantize(torch.bfloat16).dtype, torch.bfloat16)

    def test_incompatible_layout_fails_before_mutation(self) -> None:
        model = tiny_shape(1)
        original = model.layers[0].mlp.experts[0].gate_proj
        with self.assertRaisesRegex(RuntimeError, "Expected 23"):
            quantize_experts(model, "int8")
        self.assertIs(model.layers[0].mlp.experts[0].gate_proj, original)

    def test_loading_hook_runs_before_return_and_is_restored_on_failure(self) -> None:
        class Base:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return tiny_shape()

        class Model(Base):
            pass

        module = types.ModuleType("ming_tiny.modeling_lingv3_tts")
        module.LingV3TTSForConditionalGeneration = Model
        protocol = types.ModuleType("ming_tiny.task_protocol")
        protocol.configure_release_environment = lambda: None
        package = types.ModuleType("ming_tiny")
        modules = {"ming_tiny": package, module.__name__: module, protocol.__name__: protocol}
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "later loading failure"):
                with quantized_tiny_loading("int8"):
                    loaded = Model.from_pretrained("unused")
                    self.assertIsInstance(loaded.layers[0].mlp.experts[0].gate_proj, WeightOnlyLinear)
                    raise RuntimeError("later loading failure")
            self.assertNotIn("from_pretrained", Model.__dict__)
            self.assertIsInstance(Model.from_pretrained().layers[0].mlp.experts[0].gate_proj, nn.Linear)


if __name__ == "__main__":
    unittest.main()
