# SPDX-License-Identifier: Apache-2.0
"""Opt-in hook for vLLM-Ascend 0.30 W8A8_DYNAMIC linear layers."""

from collections import Counter

import torch
import torch_npu
from qwen_npu_ops.quant_matmul import WEIGHT_SHAPES, QuantMatmul
from vllm.forward_context import get_forward_context


def _decode_only():
    try:
        metadata = get_forward_context().attn_metadata
    except AssertionError:
        return False
    values = metadata.values() if isinstance(metadata, dict) else (metadata,)
    known = False
    for value in values:
        if value is None:
            continue
        prefills = getattr(value, "num_prefills", None)
        prefill_tokens = getattr(value, "num_prefill_tokens", None)
        if prefills or prefill_tokens:
            return False
        known |= prefills is not None or prefill_tokens is not None
    return known


class QuantMatmulAdapter:
    def __init__(self, payload_dir):
        self.op = QuantMatmul(payload_dir)
        self.calls = Counter()
        self.capture_hits = Counter()
        self.prepared_weights = 0

    def prepare_model(self, model):
        weights = set()
        with torch.inference_mode():
            for layer in model.modules():
                weight, scale = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
                if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
                    continue
                if weight.data_ptr() in weights or not self.op.prepare(weight, scale):
                    continue
                weights.add(weight.data_ptr())
        self.prepared_weights = len(weights)
        if not self.prepared_weights:
            raise ValueError("No model weights match the Qwen QuantMatmul payload")

    def execute(self, x, weight, scale, token):
        shape = f"{x.shape[0]},{x.shape[1]},{weight.shape[1]}"
        if _decode_only():
            output = self.op.try_run(x, weight, scale, token, output_dtype=torch.bfloat16)
            if output is not None:
                self.calls[f"custom:{shape}"] += 1
                if torch.npu.is_current_stream_capturing():
                    self.capture_hits[shape] += 1
                return output
            self.calls[f"native:unsupported:{shape}"] += 1
        else:
            self.calls[f"native:prefill_or_unknown:{shape}"] += 1
        return torch_npu.npu_quant_matmul(
            x, weight, scale, pertoken_scale=token, bias=None, output_dtype=torch.bfloat16
        )

    def snapshot(self):
        return {
            "payload_sha256": self.op.manifest["library_sha256"],
            "calls": dict(self.calls),
            "capture_hits": dict(self.capture_hits),
            "prepared_weights": self.prepared_weights,
        }

    def install(self):
        from vllm.utils.torch_utils import direct_register_custom_op

        from vllm_ascend.quantization.methods.base import PreparedLinearInput
        from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod

        original = AscendW8A8DynamicLinearMethod.apply
        if getattr(original, "_qwen_quant_matmul", False):
            raise RuntimeError("Install the Qwen QuantMatmul adapter once per worker")

        def invoke(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
            return self.execute(x, weight, scale, token)

        def fake(x, weight, scale, token):
            return x.new_empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)

        direct_register_custom_op(
            "qwen_w8a8_quant_matmul", invoke, mutates_args=[], fake_impl=fake, dispatch_key="PrivateUse1"
        )

        def apply(method, layer, x, bias=None, tp_rank=0):
            prepared = isinstance(x, PreparedLinearInput)
            output_dtype = x.output_dtype if prepared else x.dtype
            if (
                bias is not None
                or method.act_quant_type != torch.int8
                or output_dtype != torch.bfloat16
                or tuple(layer.weight.shape) not in WEIGHT_SHAPES
            ):
                return original(method, layer, x, bias, tp_rank)
            if prepared:
                quantized, token = x.quantized, x.scale
                assert token is not None
            else:
                quantized, token = torch_npu.npu_dynamic_quant(x, dst_type=method.act_quant_type)
            restore_dim = token.dim() == 2
            if restore_dim:
                quantized, token = quantized.squeeze(dim=1), token.squeeze(dim=1)
            if quantized.ndim == 2:
                output = torch.ops.vllm.qwen_w8a8_quant_matmul(quantized, layer.weight, layer.weight_scale, token)
            else:
                output = torch_npu.npu_quant_matmul(
                    quantized,
                    layer.weight,
                    layer.weight_scale,
                    pertoken_scale=token,
                    bias=None,
                    output_dtype=output_dtype,
                )
            return output.unsqueeze(dim=1) if restore_dim else output

        apply._qwen_quant_matmul = True
        AscendW8A8DynamicLinearMethod.apply = apply
