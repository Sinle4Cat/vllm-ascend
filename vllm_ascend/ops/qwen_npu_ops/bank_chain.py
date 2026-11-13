# SPDX-License-Identifier: Apache-2.0
"""Explicit direct KV-bank route; retain compact tokens and original packing."""

import torch

from vllm_ascend.ops import gdn as native

from .kv_chain import KvGdnChain


class BankKvGdnChain(KvGdnChain):
    def prepare(self, conv, state, *, max_batch):
        # Bind every actual bank before a workspace cache hit can skip work.
        # The memory-profile dummy has different geometry from the real cache.
        self.decode.prepare_bank(conv, state)
        query = conv.shape[1] - 2
        required = {(batch, query) for batch in range(1, max_batch + 1)}
        missing = required.difference(self.decode.buckets)
        if missing:
            raise ValueError(f"Missing bank Decode buckets: {sorted(missing)}")
        for key in sorted(required):
            if key not in self.workspaces:
                self.workspaces[key] = self.decode.allocate_workspace(key[0], conv.device, query=query)

    def decode_group(self, layer, qkv, z, b, a, starts, indices, accepted=None):
        if indices.ndim == 1:
            indices = indices[:, None]
        batch, width = indices.shape
        query = layer.num_spec + 1
        key = (batch, query)
        if key not in self.workspaces or width not in (1, query) or starts.numel() != batch + 1:
            raise ValueError("No bank Decode bucket covers the real sequence/page table")
        rows = qkv.shape[0]
        if rows > batch * query:
            raise ValueError("Decode token count exceeds bank bucket capacity")
        if width == 1:
            full_ids = torch.full((batch, query), -1, dtype=torch.int32, device=indices.device)
            full_ids[:, :1].copy_(indices)
        else:
            full_ids = indices.to(torch.int32).contiguous()
        if accepted is None:
            accepted = torch.ones(batch, dtype=torch.int32, device=indices.device)
        conv, state = layer.kv_cache[:2]
        metadata = self.decode.metadata_for_bank(conv, state)
        result = self.decode(
            qkv.contiguous(),
            z.contiguous(),
            b.contiguous(),
            a.contiguous(),
            native._get_packed_conv_weights(layer),
            conv,
            layer.A_log.float().contiguous(),
            layer.dt_bias.contiguous(),
            state,
            full_ids,
            starts.to(torch.int32).contiguous(),
            accepted.to(torch.int32).contiguous(),
            layer.norm.weight.contiguous(),
            metadata,
            workspace=self.workspaces[key],
        )
        self.calls[f"r22_kv_B{batch}_Q{query}"] += 1
        self.calls["r22_bank_groups"] += 1
        if torch.npu.is_current_stream_capturing():
            self.calls[f"r22_capture_B{batch}_Q{query}"] += 1
        self.calls["r22_logical_sequences"] += batch
        self.calls["r22_logical_tokens"] += rows
        return result[:rows]
