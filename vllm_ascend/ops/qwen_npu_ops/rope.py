# SPDX-License-Identifier: Apache-2.0
"""Register the R01 split-QKV, RMSNorm and MRoPE route for the Qwen KV worker."""

from collections import Counter

import torch
from qwen_npu_ops import Rope
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.patch.worker.patch_qwen3_5 import Qwen3NextAttention, _uses_multimodal_rope, native_qwen_mrope


class QwenRopeAdapter:
    def __init__(self, payload, model):
        self.op = Rope(payload, allow_experimental=True)
        self.calls = Counter()
        self.capture_hits = Counter()
        self.sites = [
            layer
            for name, layer in model.named_modules()
            if name.endswith(".self_attn") and isinstance(layer, Qwen3NextAttention) and self._supports_layer(layer)
        ]
        if len(self.sites) != 16:
            raise ValueError(f"R01 requires the Qwen TP4 attention geometry on 16 layers, found {len(self.sites)}")

        def rotate(
            qkv: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.execute(qkv, qw, kw, cache, positions)

        def fake(qkv, qw, kw, cache, positions):
            return tuple(qkv.new_empty((qkv.shape[0], width)) for width in (1536, 256, 256, 1536))

        direct_register_custom_op("qwen_rope_r01", rotate, mutates_args=[], fake_impl=fake, dispatch_key="PrivateUse1")
        for layer in self.sites:
            layer._qwen_rope_r01 = True

    @staticmethod
    def _supports_layer(layer):
        rotary = layer.rotary_emb
        return (
            _uses_multimodal_rope(layer)
            and (layer.num_heads, layer.num_kv_heads, layer.head_dim) == (6, 1, 256)
            and layer.config.rms_norm_eps == 1e-6
            and list(rotary.mrope_section) == [11, 11, 10]
            and rotary.mrope_interleaved
            and rotary.rotary_dim == 64
            and layer.attn_output_gate
        )

    def execute(self, qkv, qw, kw, cache, positions):
        rows = qkv.shape[0]
        if self.op.supports(qkv, qw, kw, cache, positions):
            result = self.op(qkv, qw, kw, cache, positions)
            self.calls[f"r01_M{rows}"] += 1
            if torch.npu.is_current_stream_capturing():
                self.capture_hits[f"M{rows}"] += 1
            return result
        self.calls[f"native_M{rows}"] += 1
        return native_qwen_mrope(
            qkv,
            qw,
            kw,
            cache,
            positions,
            num_q_heads=6,
            num_kv_heads=1,
            head_size=256,
            eps=1e-6,
            mrope_section=[11, 11, 10],
            is_interleaved=True,
            rope_dim=64,
            has_gate=True,
        )

    def snapshot(self):
        return {"sites": len(self.sites), "calls": dict(self.calls), "capture_hits": dict(self.capture_hits)}
