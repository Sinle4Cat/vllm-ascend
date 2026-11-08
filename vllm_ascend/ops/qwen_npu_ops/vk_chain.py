# SPDX-License-Identifier: Apache-2.0
"""R30 mixed prefill with native decode and persistent VK states."""

from copy import copy

import torch

from vllm_ascend.ops import gdn as native


class VkGdnChain:
    def __init__(self, prefill):
        self.prefill = prefill

    def supports(self, layer, meta, qkv, z, b, a):
        if meta.num_prefills < 1 or (meta.num_decodes == 0 and meta.spec_sequence_masks is None):
            return False
        count = int(meta.num_actual_tokens)
        if any(x.shape[0] < count for x in (qkv, z, b, a)):
            return False
        spec = meta.spec_sequence_masks is not None
        non_count = count
        if spec:
            if meta.spec_token_indx is None or meta.non_spec_token_indx is None:
                return False
            non_count = meta.non_spec_token_indx.numel()
            if meta.spec_token_indx.numel() + non_count != count or meta.spec_decode_metadata is None:
                return False
        ndecode = int(meta.num_decode_tokens) if meta.num_decodes else 0
        if meta.num_decodes and (ndecode != meta.num_decodes or meta.non_spec_decode_metadata is None):
            return False
        prefill_count = non_count - ndecode
        if prefill_count < 1:
            return False
        # Only host counts change; device lengths, state IDs and history stay intact.
        pure = copy(meta)
        pure.num_actual_tokens = prefill_count
        pure.num_decodes = pure.num_spec_decodes = 0
        pure.spec_sequence_masks = None
        return self.prefill.supports(layer, pure, *[x[:prefill_count] for x in (qkv, z, b, a)])

    @staticmethod
    def decode_group(layer, meta, qkv, z, b, a, *, speculative):
        """Use the native conv, gating, recurrence and norm without changing metadata."""
        if speculative:
            cm = meta.spec_decode_metadata.spec_causal_conv1d
            lengths = meta.spec_decode_metadata.actual_seq_lengths
            indices = meta.spec_state_indices_tensor.flatten()
            extra = dict(num_accepted_tokens=cm.num_accepted_tokens)
            query = layer.num_spec + 1
        else:
            cm = meta.non_spec_decode_metadata.causal_conv1d
            lengths = meta.non_spec_decode_metadata.actual_seq_lengths
            indices = meta.non_spec_state_indices_tensor[: meta.num_decodes]
            extra, query = {}, 1
        conv = native.causal_conv1d_update(
            qkv,
            layer.kv_cache[0],
            native._get_packed_conv_weights(layer),
            bias=layer.conv1d.bias,
            activation="silu" if layer.activation else None,
            conv_state_indices=native._normalize_causal_cache_indices(cm.cache_indices),
            query_start_loc=cm.query_start_loc,
            max_query_len=query,
            null_block_id=0,
            out=torch.empty_like(qkv),
            **extra,
        )
        q, k, v = layer.rearrange_mixed_qkv(conv)
        g, beta = native.DeviceOperator.fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        if speculative:
            extra = dict(num_accepted_tokens=cm.num_accepted_tokens.to(torch.int32))
        recurrent = getattr(layer, "qwen_recurrent", native.recurrent_gated_delta_rule)
        value = recurrent(
            native.l2norm_fwd(q).squeeze(0),
            native.l2norm_fwd(k).squeeze(0),
            v.squeeze(0),
            layer.kv_cache[1],
            g=g.squeeze(0),
            beta=beta.squeeze(0),
            scale=k.shape[-1] ** -0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=indices,
            **extra,
        )
        return layer.norm(value.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim)).view_as(z)

    def __call__(self, layer, meta, qkv, z, b, a, output):
        count = int(meta.num_actual_tokens)
        values = [x[:count] for x in (qkv, z, b, a)]
        spec = meta.spec_sequence_masks is not None
        if spec:
            selected = [x.index_select(0, meta.spec_token_indx).contiguous() for x in values]
            decoded = self.decode_group(layer, meta, *selected, speculative=True)
            output.index_copy_(0, meta.spec_token_indx, decoded)
            values = [x.index_select(0, meta.non_spec_token_indx).contiguous() for x in values]
        non_out = torch.empty_like(values[1]) if spec else output[:count]
        ndecode = int(meta.num_decode_tokens) if meta.num_decodes else 0
        if ndecode:
            decoded = self.decode_group(layer, meta, *[x[:ndecode] for x in values], speculative=False)
            non_out[:ndecode].copy_(decoded)
        ends = meta.non_spec_prefill_metadata.chunk.cu_seqlens_host
        weight = native._get_packed_conv_weights(layer)
        for seq, (start, end) in enumerate(zip(ends, ends[1:])):
            start, end = int(start) + ndecode, int(end) + ndecode
            self.prefill.run_sequence(
                *[x[start:end] for x in values],
                weight,
                layer.A_log,
                layer.dt_bias,
                layer.norm.weight,
                *layer.kv_cache[:2],
                meta.prefill_state_indices[seq : seq + 1],
                meta.prefill_has_initial_state[seq : seq + 1],
                non_out[start:end],
            )
        if spec:
            output.index_copy_(0, meta.non_spec_token_indx, non_out)
        return len(ends) - 1
