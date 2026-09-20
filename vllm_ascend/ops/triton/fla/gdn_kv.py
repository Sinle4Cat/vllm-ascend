# SPDX-License-Identifier: Apache-2.0

"""KV SSM fallback and compact adapters for the opt-in xLLM KV backend."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.layernorm import LayerNormFn
from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch


def get_xllm_kv_conv_weights(layer, weights: torch.Tensor) -> torch.Tensor:
    """Publish persistent weights only after an eager copy has executed."""
    packed = getattr(layer, "_xllm_kv_conv_weights", None)
    if packed is None:
        packed = weights.transpose(0, 1).contiguous()
        # A copy recorded in one graph has not initialized its buffer. Other
        # capture buckets must not reuse it before that graph has replayed.
        capturing = weights.device.type == "npu" and torch.npu.is_current_stream_capturing()
        if not capturing:
            layer._xllm_kv_conv_weights = packed
    return packed


@triton.jit
def _recurrent_kv_kernel(
    Q,
    K,
    V,
    G,
    Beta,
    State,
    Indices,
    Starts,
    Accepted,
    Out,
    T: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    HAS_ACCEPTED: tl.constexpr,
    INDEX_S: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    vk = tl.arange(0, BK)
    vv = tl.program_id(2) * BV + tl.arange(0, BV)
    start = tl.load(Starts + batch)
    end = tl.load(Starts + batch + 1)
    length = end - start
    accepted = 1
    if HAS_ACCEPTED:
        accepted = tl.load(Accepted + batch)
    live = (length > 0) & (length <= INDEX_S) & (start >= 0) & (end <= T)
    live = live & (accepted >= 1) & (accepted <= INDEX_S)
    slot = tl.load(Indices + batch * INDEX_S + accepted - 1, mask=live, other=0)
    live = live & (slot > 0) & (slot < N)
    state_cols = head * DK * DV + vk[:, None] * DV + vv[None, :]
    state_mask = (vk[:, None] < DK) & (vv[None, :] < DV)
    h = tl.load(State + slot * STATE_STRIDE + state_cols, mask=state_mask & live, other=0).to(tl.float32)
    key_head = head // (HV // HK)
    for token in range(start, end):
        q = tl.load(Q + (token * HK + key_head) * DK + vk, mask=(vk < DK) & live, other=0).to(tl.float32)
        k = tl.load(K + (token * HK + key_head) * DK + vk, mask=(vk < DK) & live, other=0).to(tl.float32)
        v = tl.load(V + (token * HV + head) * DV + vv, mask=(vv < DV) & live, other=0).to(tl.float32)
        g = tl.load(G + token * HV + head, mask=live, other=0).to(tl.float32)
        beta = tl.load(Beta + token * HV + head, mask=live, other=0).to(tl.float32)
        h = h * tl.exp(g)
        prediction = tl.sum(h * k[:, None], axis=0)
        delta = (v - prediction) * beta
        h = h + k[:, None] * delta[None, :]
        output = tl.sum(h * (q[:, None] * SCALE), axis=0)
        write_slot = tl.load(Indices + batch * INDEX_S + token - start, mask=live, other=0)
        write_live = live & (write_slot > 0) & (write_slot < N)
        tl.store(State + write_slot * STATE_STRIDE + state_cols, h, mask=state_mask & write_live)
        tl.store(Out + (token * HV + head) * DV + vv, output, mask=(vv < DV) & write_live)
        # MTE3 must finish reading this checkpoint's UB state before the next
        # recurrent step reuses it. Without this, a checkpoint can contain H(t+1).
        tl.debug_barrier()


def recurrent_gated_delta_rule_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Directly read/write native checkpoint IDs, with no VK materialization."""
    if state.dtype != torch.float32:
        raise ValueError("The KV backend requires --mamba-ssm-cache-dtype float32")
    t, hk, dk = query.shape
    hv, dv = value.shape[1:]
    if state.shape[1:] != (hv, dk, dv) or not state[0].is_contiguous():
        raise ValueError("Expected SSM cache [slot, value_head, key_dim, value_dim]")
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    g, beta = g.contiguous(), beta.contiguous()
    indices = ssm_state_indices.reshape(-1).contiguous()
    starts = query_start_loc.contiguous()
    accepted = starts if num_accepted_tokens is None else num_accepted_tokens.contiguous()
    out = torch.zeros_like(value)
    _recurrent_kv_kernel[(starts.numel() - 1, hv, triton.cdiv(dv, 32))](
        query,
        key,
        value,
        g,
        beta,
        state,
        indices,
        starts,
        accepted,
        out,
        T=t,
        HK=hk,
        HV=hv,
        DK=dk,
        DV=dv,
        N=state.shape[0],
        STATE_STRIDE=state.stride(0),
        HAS_ACCEPTED=num_accepted_tokens is not None,
        INDEX_S=indices.numel() // (starts.numel() - 1),
        BK=triton.next_power_of_2(dk),
        BV=32,
        SCALE=dk**-0.5 if scale is None else scale,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _pack_kernel(
    Qkv,
    Z,
    B,
    A,
    NativeConv,
    NativeSsm,
    Indices,
    Starts,
    Accepted,
    PackedQkv,
    PackedZ,
    PackedB,
    PackedA,
    Conv,
    Ssm,
    SafeAccepted,
    S: tl.constexpr,
    C: tl.constexpr,
    OUT_SIZE: tl.constexpr,
    HV: tl.constexpr,
    CONV_LEN: tl.constexpr,
    SS: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    QKV_ROW_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    CONV_STRIDE: tl.constexpr,
    SSM_STRIDE: tl.constexpr,
    MTP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(Starts + batch)
    length = tl.load(Starts + batch + 1) - start
    base_slot = tl.load(Indices + batch * S)
    accepted = 1
    if MTP:
        accepted = tl.load(Accepted + batch)
    live = (length == S) & (start >= 0) & (start + S <= T) & (base_slot > 0) & (base_slot < N)
    live = live & (accepted >= 1) & (accepted <= S)
    if tl.program_id(1) == 0:
        tl.store(SafeAccepted + batch, tl.where(live, accepted, 1))
    for token in range(S):
        # QKV/Z projection splits share a wider source row. Their channels
        # are contiguous, but the token stride is not their visible width.
        qkv = tl.load(Qkv + (start + token) * QKV_ROW_STRIDE + cols, mask=(cols < C) & live, other=0)
        z = tl.load(Z + (start + token) * Z_ROW_STRIDE + cols, mask=(cols < OUT_SIZE) & live, other=0)
        b = tl.load(B + (start + token) * HV + cols, mask=(cols < HV) & live, other=0)
        a = tl.load(A + (start + token) * HV + cols, mask=(cols < HV) & live, other=0)
        tl.store(PackedQkv + (batch * S + token) * C + cols, qkv, mask=cols < C)
        tl.store(PackedZ + (batch * S + token) * OUT_SIZE + cols, z, mask=cols < OUT_SIZE)
        tl.store(PackedB + (batch * S + token) * HV + cols, b, mask=cols < HV)
        tl.store(PackedA + (batch * S + token) * HV + cols, a, mask=cols < HV)
        slot = tl.load(Indices + batch * S + token)
        h = tl.load(NativeSsm + slot * SSM_STRIDE + cols, mask=(cols < SS) & live & (slot > 0) & (slot < N), other=0)
        tl.store(Ssm + (batch * S + token) * SS + cols, h, mask=cols < SS)
    conv = tl.load(NativeConv + base_slot * CONV_STRIDE + cols, mask=(cols < CONV_LEN * C) & live, other=0)
    tl.store(Conv + batch * CONV_LEN * C + cols, conv, mask=cols < CONV_LEN * C)


@triton.jit
def _scatter_kernel(
    NativeConv,
    NativeSsm,
    Indices,
    Starts,
    Accepted,
    Conv,
    Ssm,
    Output,
    Out,
    S: tl.constexpr,
    C: tl.constexpr,
    OUT_SIZE: tl.constexpr,
    CONV_LEN: tl.constexpr,
    SS: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    CONV_STRIDE: tl.constexpr,
    SSM_STRIDE: tl.constexpr,
    MTP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(Starts + batch)
    length = tl.load(Starts + batch + 1) - start
    base_slot = tl.load(Indices + batch * S)
    accepted = 1
    if MTP:
        accepted = tl.load(Accepted + batch)
    live = (length == S) & (start >= 0) & (start + S <= T) & (base_slot > 0) & (base_slot < N)
    live = live & (accepted >= 1) & (accepted <= S)
    conv = tl.load(Conv + batch * CONV_LEN * C + cols, mask=(cols < CONV_LEN * C) & live, other=0)
    tl.store(NativeConv + base_slot * CONV_STRIDE + cols, conv, mask=(cols < CONV_LEN * C) & live)
    for token in range(S):
        slot = tl.load(Indices + batch * S + token)
        h = tl.load(Ssm + (batch * S + token) * SS + cols, mask=(cols < SS) & live, other=0)
        tl.store(NativeSsm + slot * SSM_STRIDE + cols, h, mask=(cols < SS) & live & (slot > 0) & (slot < N))
        output = tl.load(Output + (batch * S + token) * OUT_SIZE + cols, mask=(cols < OUT_SIZE) & live, other=0)
        tl.store(Out + (start + token) * OUT_SIZE + cols, output, mask=(cols < OUT_SIZE) & live)


def _compact_decode_kv_reference(
    layer,
    ops,
    qkv,
    b,
    a,
    z,
    metadata,
    weights,
    native_conv,
    native_ssm,
    *,
    mtp: bool,
) -> torch.Tensor:
    """Unique compact slots isolate padding; masked writes leave slot zero intact."""
    if mtp:
        indices = metadata.spec_state_indices_tensor.contiguous()
        causal = metadata.spec_decode_metadata.spec_causal_conv1d
        batch, sequence = indices.shape
        conv_length = sequence + 2
    else:
        indices = metadata.non_spec_state_indices_tensor.contiguous()
        causal = metadata.non_spec_decode_metadata.causal_conv1d
        batch, sequence, conv_length = indices.numel(), 1, 3
    c = qkv.shape[-1]
    hv, d = z.shape[-2:]
    ssm_size = hv * d * d
    packed_qkv = qkv.new_empty((batch, sequence, c))
    packed_z = z.new_empty((batch, sequence, hv, d))
    packed_b = b.new_empty((batch, sequence, hv))
    packed_a = a.new_empty((batch, sequence, hv))
    conv = native_conv.new_empty((batch, conv_length, c))
    ssm = native_ssm.new_empty((batch * sequence, hv, d, d))
    accepted = torch.empty(batch, dtype=torch.int32, device=qkv.device)
    starts = causal.query_start_loc.contiguous()
    original_accepted = causal.num_accepted_tokens if mtp else starts
    block = 4096
    grid = (batch, triton.cdiv(max(ssm_size, conv_length * c, hv * d), block))
    common = dict(
        S=sequence,
        C=c,
        OUT_SIZE=hv * d,
        CONV_LEN=conv_length,
        SS=ssm_size,
        N=native_ssm.shape[0],
        T=qkv.shape[0],
        CONV_STRIDE=native_conv.stride(0),
        SSM_STRIDE=native_ssm.stride(0),
        MTP=mtp,
        BLOCK=block,
    )
    _pack_kernel[grid](
        qkv,
        z,
        b,
        a,
        native_conv,
        native_ssm,
        indices,
        starts,
        original_accepted,
        packed_qkv,
        packed_z,
        packed_b,
        packed_a,
        conv,
        ssm,
        accepted,
        HV=hv,
        QKV_ROW_STRIDE=qkv.stride(0),
        Z_ROW_STRIDE=z.stride(0),
        **common,
    )
    op_indices = torch.arange(batch, dtype=torch.int32, device=qkv.device)
    packed_weights = get_xllm_kv_conv_weights(layer, weights)
    args = (
        packed_weights,
        conv,
        layer.A_log.to(torch.float32).contiguous(),
        layer.dt_bias.to(torch.float32).contiguous(),
        ssm,
        op_indices,
        op_indices,
    )
    norm_weight = layer.norm.weight.contiguous()
    if mtp:
        _, _, _, output = ops.mega_gdn_mtp_decode(
            packed_qkv, packed_z, packed_b, packed_a, *args, accepted, norm_weight, True
        )
    else:
        _, _, _, output = ops.mega_gdn_decode(
            packed_qkv[:, 0], packed_z[:, 0], packed_b[:, 0], packed_a[:, 0], *args, norm_weight, True
        )
    out = torch.zeros_like(z)
    _scatter_kernel[grid](native_conv, native_ssm, indices, starts, original_accepted, conv, ssm, output, out, **common)
    if mtp:
        # Shape-based Python dispatch is frozen by graph capture. Uniform rows
        # use MegaMTP; shorter rows execute the native conv + direct KV fallback
        # in the same graph, with disjoint cache IDs and output positions.
        lengths = starts[1:] - starts[:-1]
        ragged_indices = torch.where(((lengths > 0) & (lengths < sequence)).unsqueeze(1), indices, 0).contiguous()
        convolved = torch.zeros_like(qkv)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            convolved,
            qkv,
            packed_weights,
            conv_state=native_conv,
            bias_opt=None,
            query_start_loc_opt=starts,
            cache_indices_opt=ragged_indices,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=original_accepted,
            activation_mode=1,
            pad_slot_id=0,
            run_mode=1,
        )
        hk = (c // d - hv) // 2
        q, k, v = convolved.split((hk * d, hk * d, hv * d), dim=-1)
        q = l2norm_fwd(q.contiguous().reshape(-1, hk, d))
        k = l2norm_fwd(k.contiguous().reshape(-1, hk, d))
        v = v.contiguous().reshape(-1, hv, d)
        g, beta = fused_gdn_gating_patch(layer.A_log, a, b, layer.dt_bias)
        fallback = recurrent_gated_delta_rule_kv(
            q,
            k,
            v,
            g.squeeze(0),
            beta.squeeze(0),
            native_ssm,
            ragged_indices,
            starts,
            original_accepted,
        )
        normalized = LayerNormFn.apply(fallback, norm_weight, None, z, 1e-6, None, True, True)
        out = out + normalized
    return out


def fused_decode_kv(layer, ops, qkv, b, a, z, metadata, weights, native_conv, native_ssm, *, mtp):
    from vllm_ascend.ops.triton.fla.gdn_native import native_decode_kv

    return native_decode_kv(
        layer, qkv, b, a, z, metadata, get_xllm_kv_conv_weights(layer, weights), native_conv, native_ssm, mtp=mtp
    )
