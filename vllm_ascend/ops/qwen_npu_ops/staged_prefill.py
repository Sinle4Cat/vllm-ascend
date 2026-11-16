# SPDX-License-Identifier: Apache-2.0
"""Prepare/H/O Prefill with native Conv, gating and normalization."""

import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

from vllm_ascend.ops import gdn as native


@contextmanager
def _selected_vendor(vendor):
    """Select an already-registered provider without adding one after init."""
    original = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    prefixes = [Path(p).resolve() for p in original.split(os.pathsep) if p]
    if vendor not in prefixes:
        raise ValueError("Activate the Chunk GDN vendor before device initialization")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys([str(vendor), *map(str, prefixes)]))
    try:
        yield
    finally:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = original


def _plain(value):
    value = value.contiguous()
    return value.clone(memory_format=torch.contiguous_format) if value.storage_offset() else value


class ThreeStagePrefill:
    backend = "three_stage"

    def __init__(self, identity, *, ssm_layout="KV"):
        from qwen_npu_ops import ChunkGdnH, ChunkGdnO, ChunkGdnPrepare

        if ssm_layout not in ("KV", "VK"):
            raise ValueError("Select KV or VK Prefill state layout")
        self.ssm_layout = ssm_layout
        self.identity_path = Path(identity).resolve()
        selected = json.loads(self.identity_path.read_text())
        if selected.get("schema") != "qwen-chunk-gdn-runtime-identity/1":
            raise ValueError("Three-stage Prefill requires a Chunk GDN runtime identity")
        if selected.get("entry_points") != ["full", "prepare", "h", "o"]:
            raise ValueError("Rebuild the provider with independent Prepare/H/O entries")
        self.vendor = Path(selected["vendor"]).resolve()
        with _selected_vendor(self.vendor):
            self.prepare_op, self.h_op, self.o_op = [
                cls(identity, allow_experimental=True) for cls in (ChunkGdnPrepare, ChunkGdnH, ChunkGdnO)
            ]
        self.calls = Counter()
        self.shapes = Counter()

    def prepare(self, device):
        # Workspaces follow the model stream inside each explicit runtime API.
        return None

    @staticmethod
    def _boundaries(meta, rows):
        chunk = getattr(meta.non_spec_prefill_metadata, "chunk", None)
        cu = None if chunk is None else getattr(chunk, "cu_seqlens_host", None)
        if cu is None or len(cu) != meta.num_prefills + 1:
            raise ValueError("Three-stage Prefill requires exact host sequence boundaries")
        ends = [int(value) for value in cu]
        if ends[0] != 0 or ends[-1] != rows or any(hi <= lo for lo, hi in zip(ends, ends[1:])):
            raise ValueError("Prefill boundaries must cover the compact ragged token group")
        return cu

    @staticmethod
    def supports(layer, meta, qkv, z, b, a):
        if meta.num_prefills < 1 or meta.num_decodes or meta.num_spec_decodes:
            return False
        if meta.spec_sequence_masks is not None:
            return False
        return (
            qkv.shape[0] >= int(meta.num_actual_tokens) > 0
            and tuple(qkv.shape[1:]) == (2560,)
            and tuple(z.shape[1:]) == (12, 128)
            and tuple(b.shape[1:]) == (12,)
            and a.shape == b.shape
            and all(value.dtype == torch.bfloat16 for value in (qkv, z, b, a))
        )

    def forward(self, q, k, v, g, beta, initial, cu):
        """Keep the qualified composition's cast boundaries and KV H/O ABI."""
        if initial.dtype != torch.float32:
            raise ValueError("Three-stage model Prefill requires FP32 SSM state")
        q, k, v = [_plain(value) for value in (q, k, v)]
        g = _plain(g.transpose(1, 2).float())
        beta = _plain(beta.transpose(1, 2).float())
        initial = _plain(initial)
        gc, w, u, _matrix, qh, kh, _qr, _kr, _be = self.prepare_op(
            q, k, v, g, beta, cu_seqlens=cu, use_qk_l2norm=False, output_a=True
        )
        history, v_new, final = self.h_op(kh, w, u, gc, initial_state=initial, cu_seqlens=cu, state_layout="KV")
        head = self.o_op(qh, kh, v_new, history, gc, cu_seqlens=cu, scale=k.shape[-1] ** -0.5)
        self.calls["prepare"] += 1
        self.calls["h"] += 1
        self.calls["o"] += 1
        self.shapes[str(tuple(q.shape))] += 1
        return head.transpose(1, 2).contiguous(), final

    def run_group(self, layer, meta, qkv, z, b, a, output):
        """Consume only Prefill rows after the mixed group's Decode prefix."""
        cu = self._boundaries(meta, qkv.shape[0])
        conv, state = layer.kv_cache[:2]
        if (
            layer.tp_size != 4
            or layer.conv1d.bias is not None
            or not layer.norm.norm_before_gate
            or layer.norm.eps != 1e-6
            or state.dtype != torch.float32
        ):
            raise ValueError("Unsupported Qwen TP4 Prefill/state contract")
        conv_out = native.causal_conv1d_fn(
            qkv,
            native._get_packed_conv_weights(layer),
            layer.conv1d.bias,
            conv_states=conv,
            query_start_loc=meta.prefill_query_start_loc,
            cache_indices=native._normalize_causal_cache_indices(meta.prefill_state_indices),
            has_initial_state=meta.prefill_has_initial_state,
            activation="silu" if layer.activation else None,
            pad_slot_id=native.PAD_SLOT_ID,
            null_block_id=0,
        )
        q, k, v = layer.rearrange_mixed_qkv(conv_out)
        g, beta = native.DeviceOperator.fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        initial = native.gather_ssm_states(state, meta.prefill_state_indices, meta.prefill_has_initial_state)
        if self.ssm_layout == "VK":
            initial = initial.transpose(-1, -2).contiguous()
        value, final = self.forward(
            native.l2norm_fwd(q).transpose(1, 2).contiguous(),
            native.l2norm_fwd(k).transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            g.float().contiguous(),
            beta.to(v.dtype).contiguous(),
            initial,
            cu,
        )
        if self.ssm_layout == "VK":
            final = final.transpose(-1, -2).contiguous()
        native.scatter_ssm_states_(state, meta.prefill_state_indices, final.contiguous())
        output.copy_(layer.norm(value.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim)).view_as(z))
        self.calls["groups"] += 1
        self.calls["sequences"] += int(meta.num_prefills)
        return int(meta.num_prefills)

    def __call__(self, layer, meta, qkv, z, b, a, output, weight):
        count = int(meta.num_actual_tokens)
        return self.run_group(layer, meta, *[value[:count] for value in (qkv, z, b, a)], output[:count])

    def status(self):
        return dict(
            backend=self.backend,
            identity=str(self.identity_path),
            identity_sha256=hashlib.sha256(self.identity_path.read_bytes()).hexdigest(),
            state_layout=self.ssm_layout,
            calls=dict(self.calls),
            shapes=dict(self.shapes),
            runtime=self.prepare_op.status(),
        )
