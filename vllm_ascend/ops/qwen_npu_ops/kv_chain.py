"""Persistent KV dispatch: mixed/ragged Prefill uses R30, Decode uses R22."""

from collections import Counter

import torch

from vllm_ascend.ops import gdn as native

from .state_bridge import gather, prepare, scatter


class KvGdnChain:
    def __init__(self, decode, prefill):
        self.decode, self.prefill = decode, prefill
        self.bridges, self.workspaces = {}, {}
        self.calls = Counter()

    def prepare(self, conv, state, *, max_batch):
        query = conv.shape[1] - 2
        required = {(batch, query) for batch in range(1, max_batch + 1)}
        missing = required.difference(self.decode.buckets)
        if missing:
            raise ValueError(f"Missing R22 decode buckets for configured concurrency: {sorted(missing)}")
        for batch in range(1, max_batch + 1):
            key = (batch, query)
            if key in self.bridges:
                continue
            ids = torch.empty(key, dtype=torch.int32, device=conv.device)
            self.bridges[key] = prepare(conv, state, ids, ssm_layout="KV")
            self.workspaces[key] = self.decode.allocate_workspace(batch, conv.device, query=query)

    def decode_group(self, layer, qkv, z, b, a, starts, indices, accepted=None):
        """Keep compact token order, real lengths and the complete history table.

        R22 receives exactly sum(lengths) rows. Its kernels already address
        compact rows using QSTART; no per-sequence padding/copy is required.
        """
        if indices.ndim == 1:
            indices = indices[:, None]
        batch, width = indices.shape
        query = layer.num_spec + 1
        key = (batch, query)
        if key not in self.bridges or width not in (1, query) or starts.numel() != batch + 1:
            raise ValueError("No R22 bucket covers the full decode sequence/page table")
        buffers = self.bridges[key]
        rows = qkv.shape[0]
        if rows > key[0] * query:
            raise ValueError("Decode token count exceeds its R22 bucket")
        if width == 1:
            full_ids = torch.full((batch, query), -1, dtype=torch.int32, device=indices.device)
            full_ids[:, :1].copy_(indices)
        else:
            full_ids = indices.to(torch.int32).contiguous()
        if accepted is None:
            accepted = torch.ones(batch, dtype=torch.int32, device=indices.device)
        conv, state = layer.kv_cache[:2]
        gather(conv, state, full_ids, buffers)
        result = self.decode(
            qkv.contiguous(),
            z.contiguous(),
            b.contiguous(),
            a.contiguous(),
            native._get_packed_conv_weights(layer),
            buffers["conv"],
            layer.A_log.float().contiguous(),
            layer.dt_bias.contiguous(),
            buffers["ssm"],
            buffers["local"],
            starts.to(torch.int32).contiguous(),
            accepted.to(torch.int32).contiguous(),
            layer.norm.weight.contiguous(),
            buffers["capacity"],
            workspace=self.workspaces[key],
        )
        scatter(conv, state, full_ids, buffers)
        self.calls[f"r22_kv_B{key[0]}_Q{query}"] += 1
        if torch.npu.is_current_stream_capturing():
            self.calls[f"r22_capture_B{batch}_Q{query}"] += 1
        self.calls["r22_logical_sequences"] += batch
        self.calls["r22_logical_tokens"] += rows
        return result[:rows]

    def __call__(self, layer, meta, qkv, z, b, a, output):
        if getattr(layer, "_qwen_ssm_layout", None) != "KV":
            raise RuntimeError("KV chain requires a persistent KV allocation contract")
        count = int(meta.num_actual_tokens)
        qkv, z, b, a = [x[:count] for x in (qkv, z, b, a)]
        spec = meta.spec_sequence_masks is not None
        mixed = spec and (meta.num_prefills > 0 or meta.num_decodes > 0)
        if spec:
            token_ids = meta.spec_token_indx if mixed else None
            values = [x.index_select(0, token_ids) if mixed else x for x in (qkv, z, b, a)]
            cm = meta.spec_decode_metadata.spec_causal_conv1d
            result = self.decode_group(
                layer, *values, cm.query_start_loc, meta.spec_state_indices_tensor, cm.num_accepted_tokens
            )
            if mixed:
                output.index_copy_(0, token_ids, result)
            else:
                output[:count].copy_(result)
            self.calls["r22_spec_groups"] += 1
        if spec and not mixed:
            return
        token_ids = meta.non_spec_token_indx if mixed else None
        values = [x.index_select(0, token_ids) if mixed else x for x in (qkv, z, b, a)]
        non_out = torch.zeros_like(values[1])
        ndecode = int(meta.num_decode_tokens) if meta.num_decodes > 0 else 0
        if ndecode:
            cm = meta.non_spec_decode_metadata.causal_conv1d
            result = self.decode_group(
                layer,
                *[x[:ndecode] for x in values],
                cm.query_start_loc,
                meta.non_spec_state_indices_tensor[: meta.num_decodes],
            )
            non_out[:ndecode].copy_(result)
            self.calls["r22_single_groups"] += 1
        if meta.num_prefills:
            chunk = meta.non_spec_prefill_metadata.chunk
            ends = None if chunk is None else getattr(chunk, "cu_seqlens_host", None)
            if ends is None or len(ends) != meta.num_prefills + 1:
                raise ValueError("R30 mixed prefill requires exact host sequence boundaries")
            ends = [int(x) for x in ends]
            if ends[0] != 0 or ends[-1] + ndecode != values[0].shape[0]:
                raise ValueError("R30 prefill boundaries do not cover the non-spec token group")
            weight = native._get_packed_conv_weights(layer)
            for seq, (lo, hi) in enumerate(zip(ends, ends[1:])):
                if hi < lo:
                    raise ValueError("Prefill boundaries must be monotonic")
                if hi == lo:
                    continue
                lo, hi = lo + ndecode, hi + ndecode
                self.prefill.run_sequence(
                    *[x[lo:hi] for x in values],
                    weight,
                    layer.A_log,
                    layer.dt_bias,
                    layer.norm.weight,
                    *layer.kv_cache[:2],
                    meta.prefill_state_indices[seq : seq + 1],
                    meta.prefill_has_initial_state[seq : seq + 1],
                    non_out[lo:hi],
                )
                self.calls["r30_mixed_sequences"] += 1
                self.calls["r30_mixed_tokens"] += hi - lo
        if mixed:
            output.index_copy_(0, token_ids, non_out)
            self.calls["mixed_batches"] += 1
        else:
            output[:count].copy_(non_out)
