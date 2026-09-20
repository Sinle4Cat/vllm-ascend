# SPDX-License-Identifier: Apache-2.0
"""Direct native-bank GDN SDK adapter; metadata remains on the device."""

import torch
from vllm.logger import init_logger

logger = init_logger("vllm_ascend.ops.gdn")


def native_decode_kv(layer, qkv, b, a, z, metadata, packed_weights, native_conv, native_ssm, *, mtp):
    if mtp:
        indices = metadata.spec_state_indices_tensor.contiguous()
        causal = metadata.spec_decode_metadata.spec_causal_conv1d
        accepted = causal.num_accepted_tokens.contiguous()
    else:
        indices = metadata.non_spec_state_indices_tensor.reshape(-1, 1).contiguous()
        causal = metadata.non_spec_decode_metadata.causal_conv1d
        accepted = torch.ones_like(indices[:, 0])
    return _native_decode_kv(
        layer,
        qkv,
        b,
        a,
        z,
        packed_weights,
        native_conv,
        native_ssm,
        indices,
        causal.query_start_loc,
        accepted,
    )


def _native_decode_kv(layer, qkv, b, a, z, packed_weights, native_conv, native_ssm, indices, starts, accepted):
    import native_ops_lib

    logger.warning_once("Native SDK GDN decode enabled: direct banks, device lengths and checkpoints")
    return native_ops_lib.mega_gdn_native_decode(
        qkv,
        z,
        b.contiguous(),
        a.contiguous(),
        packed_weights,
        native_conv,
        layer.A_log.float().contiguous(),
        layer.dt_bias.float().contiguous(),
        native_ssm,
        indices,
        starts.contiguous(),
        accepted,
        layer.norm.weight.contiguous(),
    )


def native_prefill_kv(layer, qkv, b, a, z, metadata, packed_weights, native_conv, native_ssm):
    import native_ops_lib

    write = metadata.prefill_state_indices.to(torch.int32).contiguous()
    read = torch.where(metadata.prefill_has_initial_state.to(torch.bool), write, -1).contiguous()
    starts = metadata.prefill_query_start_loc.to(torch.int32).contiguous()
    lower, full, minus = layer._get_xllm_masks(qkv.device, qkv.dtype)
    host_starts = metadata.non_spec_prefill_metadata.chunk.cu_seqlens_host
    matrices = sum((end - start + 127) // 128 for start, end in zip(host_starts, host_starts[1:]))
    matrices *= layer.num_v_heads // layer.tp_size
    logger.warning_once("Native SDK GDN prefill enabled: direct banks, three-row logical history")
    return native_ops_lib.mega_gdn_native_prefill(
        qkv.contiguous(),
        b.contiguous(),
        a.contiguous(),
        z.contiguous(),
        packed_weights,
        native_conv,
        layer.A_log.float().contiguous(),
        layer.dt_bias.float().contiguous(),
        read,
        write,
        read,
        write,
        native_ssm,
        lower,
        full,
        minus,
        starts,
        layer.norm.weight.contiguous(),
        matrices,
    )


def native_mixed_kv(layer, qkv, b, a, z, metadata, packed_weights, native_conv, native_ssm):
    """Fuse each phase of a mixed batch using the allocator's original banks.

    The builder stably partitions token indices into speculative/non-speculative
    rows. Within non-speculative rows, ordinary decodes precede prefills, whose
    query starts are already rebased. No state bank is gathered or duplicated.
    """
    logger.warning_once("Native SDK GDN mixed prefill/decode enabled: direct banks and token-index merge")
    inputs = (qkv, b, a, z)
    has_spec = metadata.num_spec_decodes > 0
    if has_spec:
        non_spec_inputs = tuple(x.index_select(0, metadata.non_spec_token_indx) for x in inputs)
        spec_inputs = tuple(x.index_select(0, metadata.spec_token_indx) for x in inputs)
        spec_out = native_decode_kv(
            layer,
            *spec_inputs,
            metadata,
            packed_weights,
            native_conv,
            native_ssm,
            mtp=True,
        )
    else:
        non_spec_inputs = inputs

    decode_tokens = metadata.num_decode_tokens
    prefill_inputs = tuple(x[decode_tokens:] for x in non_spec_inputs)
    prefill_out = native_prefill_kv(
        layer,
        *prefill_inputs,
        metadata,
        packed_weights,
        native_conv,
        native_ssm,
    )
    if decode_tokens:
        decode_inputs = tuple(x[:decode_tokens] for x in non_spec_inputs)
        indices = metadata.non_spec_state_indices_tensor[: metadata.num_decodes].reshape(-1, 1).contiguous()
        starts = metadata.non_spec_decode_metadata.causal_conv1d.query_start_loc[: metadata.num_decodes + 1]
        decode_out = _native_decode_kv(
            layer,
            *decode_inputs,
            packed_weights,
            native_conv,
            native_ssm,
            indices,
            starts,
            torch.ones_like(indices[:, 0]),
        )
        non_spec_out = torch.cat((decode_out, prefill_out), dim=0)
    else:
        non_spec_out = prefill_out
    if not has_spec:
        return non_spec_out
    output = torch.empty_like(z)
    output.index_copy_(0, metadata.spec_token_indx, spec_out)
    output.index_copy_(0, metadata.non_spec_token_indx, non_spec_out)
    return output
