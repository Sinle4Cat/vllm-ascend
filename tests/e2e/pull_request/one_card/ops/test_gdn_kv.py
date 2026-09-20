# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fla.gdn_kv import (
    fused_decode_kv,
    recurrent_gated_delta_rule_kv,
)


@pytest.mark.parametrize("length", [3, 128])
@pytest.mark.parametrize("has_initial", [False, True])
def test_prefill_uses_minimal_history_preserving_mtp_rows(length, has_initial):
    import custom_ops_lib

    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

    torch.manual_seed(20260918)
    hk, hv, d, slots = 4, 12, 128, 10
    channels = (2 * hk + hv) * d
    tokens = 2 * length + 3
    qkv = (torch.randn(tokens, channels) * 0.1).bfloat16().npu()
    b = torch.randn(tokens, hv).bfloat16().npu()
    a = torch.randn_like(b)
    z = (torch.randn(tokens, hv, d) * 0.1).bfloat16().npu()
    weights = (torch.randn(channels, 4) * 0.1).bfloat16().npu()
    conv = (torch.randn(slots, 6, channels) * 0.1).bfloat16().npu()
    state = (torch.randn(slots, hv, d, d) * 0.02).npu()
    before_conv, before_state = conv.clone(), state.clone()
    ids = torch.tensor([7, 2], dtype=torch.int32).npu()
    starts = torch.tensor([0, length, tokens], dtype=torch.int32).npu()
    initial = torch.tensor([has_initial, not has_initial], dtype=torch.bool).npu()
    layer = SimpleNamespace(
        num_v_heads=hv,
        tp_size=1,
        A_log=torch.zeros(hv).npu(),
        dt_bias=torch.zeros(hv).npu(),
        norm=SimpleNamespace(weight=torch.ones(d, dtype=torch.bfloat16).npu()),
        _get_xllm_masks=AscendGatedDeltaNetAttention._get_xllm_masks,
    )
    metadata = SimpleNamespace(
        prefill_state_indices=ids,
        prefill_has_initial_state=initial,
        prefill_query_start_loc=starts,
        non_spec_prefill_metadata=SimpleNamespace(chunk=SimpleNamespace(cu_seqlens_host=(0, length, tokens))),
    )
    compact_conv = conv[ids.long(), :3].contiguous()
    compact_state = state[ids.long()].clone()
    write = torch.arange(2, dtype=torch.int32, device=qkv.device)
    read = torch.where(initial, write, torch.full_like(write, -1))
    matrices = ((length + 127) // 128 + (length + 3 + 127) // 128) * hv
    masks = layer._get_xllm_masks(qkv.device, qkv.dtype)
    for chunk in range(2):
        expected = custom_ops_lib.mega_gdn_prefill_op(
            qkv,
            b,
            a,
            z,
            weights.T.contiguous(),
            compact_conv,
            layer.A_log,
            layer.dt_bias,
            read,
            write,
            read,
            write,
            compact_state,
            *masks,
            starts,
            layer.norm.weight,
            matrices,
        )
        actual = AscendGatedDeltaNetAttention._xllm_prefill(
            layer,
            qkv,
            b,
            a,
            z,
            metadata,
            weights,
            conv,
            state,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv[ids.long(), :3], compact_conv, rtol=0, atol=0)
        torch.testing.assert_close(state[ids.long()], compact_state, rtol=0, atol=0)
        torch.testing.assert_close(conv[:, 3:], before_conv[:, 3:], rtol=0, atol=0)
        inactive = torch.tensor([0, 1, 3, 4, 5, 6, 8, 9], device=qkv.device)
        torch.testing.assert_close(conv[inactive], before_conv[inactive], rtol=0, atol=0)
        torch.testing.assert_close(state[inactive], before_state[inactive], rtol=0, atol=0)
        assert torch.isfinite(actual).all() and torch.isfinite(state).all()
        initial.fill_(True)
        read = write


def _reference(q, k, v, g, beta, state, indices, starts, accepted):
    state = state.clone()
    out = torch.zeros_like(v)
    groups = v.shape[1] // q.shape[1]
    for batch in range(indices.shape[0]):
        start, end = starts[batch : batch + 2].tolist()
        if end == start or indices[batch, 0] == 0:
            continue
        h = state[indices[batch, accepted[batch] - 1]].clone()
        for token in range(start, end):
            kt = k[token].repeat_interleave(groups, 0).float()
            qt = q[token].repeat_interleave(groups, 0).float() * q.shape[-1] ** -0.5
            h *= g[token].float().exp()[:, None, None]
            prediction = (h * kt[:, :, None]).sum(1)
            delta = (v[token].float() - prediction) * beta[token].float()[:, None]
            h += kt[:, :, None] * delta[:, None, :]
            out[token] = (h * qt[:, :, None]).sum(1).to(v.dtype)
            state[indices[batch, token - start]] = h
    return out, state


@pytest.mark.parametrize("lengths", [(4, 4, 4, 4), (4, 2, 1, 0), (1, 1, 0, 0)])
@pytest.mark.parametrize("dims", [(128, 128), (32, 64)])
def test_direct_kv_recurrence_checkpoints_ragged_and_padding(lengths, dims):
    torch.manual_seed(20260918)
    dk, dv = dims
    hk, hv, n = 1, 2, 40
    s = max(lengths)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    indices = torch.tensor([list(range(3 + b * 8, 3 + b * 8 + s)) for b in range(4)], dtype=torch.int32)
    indices[torch.tensor(lengths) == 0] = 0
    accepted = torch.tensor([min(s, b + 1) for b in range(4)], dtype=torch.int32)
    t = sum(lengths)
    q = torch.randn(t, hk, dk, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(t, hv, dv, dtype=torch.bfloat16) * 0.1
    g = torch.full((t, hv), -0.2, dtype=torch.bfloat16)
    beta = torch.full((t, hv), 0.4, dtype=torch.bfloat16)
    state = torch.randn(n, hv, dk, dv) * 0.02
    expected_out, expected_state = _reference(q, k, v, g, beta, state, indices, starts, accepted)
    args = [x.npu() for x in (q, k, v, g, beta, state, indices, starts, accepted)]
    actual = recurrent_gated_delta_rule_kv(*args)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().float(), expected_out.float(), rtol=0.005, atol=2e-5)
    torch.testing.assert_close(args[5].cpu(), expected_state, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(args[5][0].cpu(), state[0], rtol=0, atol=0)

    if lengths == (4, 4, 4, 4) and dims == (128, 128):
        # Reused UB state previously overwrote intermediate checkpoints with
        # the next step's state. One launch alone did not reliably expose it.
        for _ in range(16):
            args[5].copy_(state.npu())
            actual = recurrent_gated_delta_rule_kv(*args)
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu().float(), expected_out.float(), rtol=0.005, atol=2e-5)
            torch.testing.assert_close(args[5].cpu(), expected_state, rtol=2e-5, atol=2e-6)


def test_recurrent_kv_graph_replay_ragged_and_padding():
    torch.manual_seed(20260918)
    q = (torch.randn(16, 1, 32) * 0.1).bfloat16()
    k = torch.randn_like(q) * 0.1
    v = (torch.randn(16, 2, 64) * 0.1).bfloat16()
    g = torch.full((16, 2), -0.2, dtype=torch.bfloat16)
    beta = torch.full((16, 2), 0.4, dtype=torch.bfloat16)
    state = torch.randn(40, 2, 32, 64) * 0.02
    indices = torch.tensor([list(range(3 + b * 8, 7 + b * 8)) for b in range(4)], dtype=torch.int32)
    starts = torch.arange(5, dtype=torch.int32) * 4
    accepted = torch.ones(4, dtype=torch.int32)
    args = [x.npu() for x in (q, k, v, g, beta, state, indices, starts, accepted)]
    for _ in range(3):
        recurrent_gated_delta_rule_kv(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream()):
        captured = recurrent_gated_delta_rule_kv(*args)
    torch.npu.synchronize()
    for lengths in ((4, 2, 1, 0), (1, 0, 0, 0), (4, 4, 4, 4)):
        changed_indices = indices + 1
        changed_indices[torch.tensor(lengths) == 0] = 0
        changed_starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        changed_accepted = torch.tensor([4, 3, 2, 1], dtype=torch.int32)
        expected_out, expected_state = _reference(
            q, k, v, g, beta, state, changed_indices, changed_starts, changed_accepted
        )
        for device, host in zip(args[5:], (state, changed_indices, changed_starts, changed_accepted)):
            device.copy_(host.npu())
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured.cpu().float(), expected_out.float(), rtol=0.005, atol=2e-5)
        torch.testing.assert_close(args[5].cpu(), expected_state, rtol=2e-5, atol=2e-6)


def _fused_case(sequence=4, hk=1, hv=2, batch=4, strided=False):
    import custom_ops_lib

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    init_device_properties_triton()

    torch.manual_seed(20260918)
    d, n = 128, max(64, batch * 11 + 10)
    c = (hv + hk * 2) * d
    indices = torch.tensor(
        [[7 + b * 11 + 2 * t for t in range(sequence)] for b in range(batch)], dtype=torch.int32
    ).npu()
    starts = (torch.arange(batch + 1, dtype=torch.int32) * sequence).npu()
    accepted = torch.tensor([min(sequence, row % sequence + 1) for row in range(batch)], dtype=torch.int32).npu()
    causal = SimpleNamespace(query_start_loc=starts, num_accepted_tokens=accepted)
    meta = SimpleNamespace(
        spec_state_indices_tensor=indices,
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=causal),
        non_spec_state_indices_tensor=indices[:, 0].contiguous(),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=causal),
    )
    layer = SimpleNamespace(
        A_log=torch.zeros(hv).npu(),
        dt_bias=torch.zeros(hv).npu(),
        norm=SimpleNamespace(weight=torch.ones(d, dtype=torch.bfloat16).npu()),
    )
    qkv = (torch.randn(batch * sequence, c) * 0.1).bfloat16().npu()
    z = (torch.randn(batch * sequence, hv, d) * 0.1).bfloat16().npu()
    if strided:
        combined = torch.cat((qkv, z.flatten(1)), dim=-1)
        qkv, flat_z = combined.split((c, hv * d), dim=-1)
        z = flat_z.reshape_as(z)
    b = torch.randn(batch * sequence, hv).bfloat16().npu()
    a = torch.randn_like(b)
    weights = (torch.randn(c, 4) * 0.1).bfloat16().npu()
    conv = (torch.randn(n, sequence + 2, c) * 0.1).bfloat16().npu()
    state = (torch.randn(n, hv, d, d) * 0.02).npu()
    if strided:
        # Native banks can share storage with other state fields. Preserve the
        # physical slot stride and all gaps; never materialize a contiguous bank.
        conv_storage = torch.full((n, conv[0].numel() + 128), 0.25, dtype=conv.dtype, device=conv.device)
        conv_view = conv_storage[:, : conv[0].numel()].view_as(conv)
        conv_view.copy_(conv)
        conv = conv_view
        state_storage = torch.full((n, state[0].numel() + 128), 0.25, dtype=state.dtype, device=state.device)
        state_view = state_storage[:, : state[0].numel()].view_as(state)
        state_view.copy_(state)
        state = state_view
    return layer, custom_ops_lib, qkv, b, a, z, meta, weights, conv, state


@pytest.mark.parametrize("sequence", [1, 4])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_b4_matches_vk_operator_and_preserves_other_slots(sequence, strided):
    args = _fused_case(sequence, strided=strided)
    layer, ops, qkv, b, a, z, meta, weights, conv, state = args
    conv_before, state_before = conv.clone(), state.clone()
    batch, hv, d = 4, z.shape[1], z.shape[2]
    indices = meta.spec_state_indices_tensor.flatten().long()
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch

    # Use the actual framework convolution + gating + recurrent operators.
    # The old SDK MegaMTP oracle uses FP32 normalized Q/K and BF16 g, which
    # differs from the model's BF16 Q/K and FP32 g contract.
    conv_work = conv.clone()
    vk_work = state.transpose(-1, -2).contiguous()
    convolved = torch.empty_like(qkv)
    causal = meta.spec_decode_metadata.spec_causal_conv1d
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        convolved,
        qkv.contiguous(),
        weights.T.contiguous(),
        conv_state=conv_work,
        bias_opt=None,
        query_start_loc_opt=causal.query_start_loc,
        cache_indices_opt=meta.spec_state_indices_tensor[:, 0].contiguous(),
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=causal.num_accepted_tokens if sequence > 1 else None,
        activation_mode=1,
        pad_slot_id=0,
        run_mode=1,
    )
    hk = (qkv.shape[1] // d - hv) // 2
    q, k, value = convolved.split((hk * d, hk * d, hv * d), dim=-1)
    q = l2norm_fwd(q.reshape(-1, hk, d).contiguous())
    k = l2norm_fwd(k.reshape(-1, hk, d).contiguous())
    value = value.reshape(-1, hv, d).contiguous()
    g, beta = fused_gdn_gating_patch(layer.A_log, a, b, layer.dt_bias)
    readout = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=q,
        key=k,
        value=value,
        state=vk_work,
        g=g.squeeze(0),
        beta=beta.squeeze(0),
        scale=d**-0.5,
        actual_seq_lengths=torch.cat((causal.query_start_loc[:1], torch.diff(causal.query_start_loc))),
        ssm_state_indices=meta.spec_state_indices_tensor.flatten(),
        num_accepted_tokens=causal.num_accepted_tokens if sequence > 1 else None,
    )
    value = readout.float()
    expected = (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        * layer.norm.weight.float()
        * torch.nn.functional.silu(z.float())
    ).bfloat16()
    vk_work = vk_work.index_select(0, indices)
    conv_work = conv_work.index_select(0, indices.reshape(batch, sequence)[:, 0])
    actual = fused_decode_kv(*args, mtp=sequence > 1)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float(), expected.reshape_as(z).float(), rtol=0.01, atol=1e-4)
    torch.testing.assert_close(state.index_select(0, indices), vk_work.transpose(-1, -2), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(conv.index_select(0, indices.reshape(batch, sequence)[:, 0]), conv_work, rtol=0, atol=0)
    selected = torch.zeros(state.shape[0], dtype=torch.bool)
    selected[indices.cpu()] = True
    torch.testing.assert_close(state.cpu()[~selected], state_before.cpu()[~selected], rtol=0, atol=0)
    selected.zero_()
    selected[indices.cpu().reshape(batch, sequence)[:, 0]] = True
    torch.testing.assert_close(conv.cpu()[~selected], conv_before.cpu()[~selected], rtol=0, atol=0)
    if strided:
        for bank in (conv, state):
            span = bank.as_strided(((bank.shape[0] - 1) * bank.stride(0) + bank[0].numel(),), (1,))
            for row in range(bank.shape[0] - 1):
                gap = span[row * bank.stride(0) + bank[0].numel() : (row + 1) * bank.stride(0)]
                torch.testing.assert_close(gap, torch.full_like(gap, 0.25), rtol=0, atol=0)


@pytest.mark.parametrize("sequence", [1, 4])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_graph_replay_changed_indices_accepted_and_padding(sequence, strided):
    args = _fused_case(sequence, strided=strided)
    layer, ops, qkv, b, a, z, meta, weights, conv, state = args
    for _ in range(3):
        fused_decode_kv(*args, mtp=sequence > 1)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream):
        captured = fused_decode_kv(*args, mtp=sequence > 1)
    torch.npu.synchronize()
    for live, accepted in ((2, 1), (1, sequence), (4, min(2, sequence))):
        ids_cpu = torch.tensor([[2 + b * 11 + t for t in range(sequence)] for b in range(4)], dtype=torch.int32)
        ids_cpu[live:] = 0
        meta.spec_state_indices_tensor.copy_(ids_cpu.npu())
        meta.non_spec_state_indices_tensor.copy_(ids_cpu[:, 0].contiguous().npu())
        causal = meta.spec_decode_metadata.spec_causal_conv1d
        causal.query_start_loc.copy_((torch.arange(5, dtype=torch.int32).clamp_max(live) * sequence).npu())
        causal.num_accepted_tokens.fill_(accepted)
        before_conv, before_state = conv.clone(), state.clone()
        expected_conv, expected_state = conv.clone(), state.clone()
        expected = fused_decode_kv(
            layer, ops, qkv, b, a, z, meta, weights, expected_conv, expected_state, mtp=sequence > 1
        )
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        touched = torch.zeros(state.shape[0], dtype=torch.bool)
        touched[ids_cpu[:live].flatten().long()] = True
        torch.testing.assert_close(state.cpu()[~touched], before_state.cpu()[~touched], rtol=0, atol=0)
        touched.zero_()
        touched[ids_cpu[:live, 0].long()] = True
        torch.testing.assert_close(conv.cpu()[~touched], before_conv.cpu()[~touched], rtol=0, atol=0)
        assert torch.count_nonzero(captured[live * sequence :]).item() == 0


def test_mtp_graph_replay_changes_request_count_and_sequence_lengths():
    args = _fused_case(4, hk=4, hv=12, strided=True)
    layer, ops, qkv, b, a, z, meta, weights, conv, state = args
    causal = meta.spec_decode_metadata.spec_causal_conv1d
    # One captured token-sized graph must cover both B1*S4 and B4*S1.
    causal.query_start_loc.copy_(torch.tensor([0, 4, 4, 4, 4], dtype=torch.int32).npu())
    for _ in range(3):
        fused_decode_kv(*args, mtp=True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream()):
        captured = fused_decode_kv(*args, mtp=True)
    torch.npu.synchronize()
    for lengths in ((1, 1, 1, 1), (4, 2, 1, 0), (4, 4, 0, 0), (1, 0, 0, 0), (4, 4, 4, 4)):
        starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        causal.query_start_loc.copy_(starts.npu())
        causal.num_accepted_tokens.copy_(torch.tensor([4, 3, 2, 1], dtype=torch.int32).npu())
        ids = torch.tensor([[2 + row * 11 + token for token in range(4)] for row in range(4)], dtype=torch.int32)
        ids[torch.tensor(lengths) == 0] = 0
        meta.spec_state_indices_tensor.copy_(ids.npu())
        before_conv, before_state = conv.clone(), state.clone()
        expected_conv, expected_state = conv.clone(), state.clone()
        expected = fused_decode_kv(layer, ops, qkv, b, a, z, meta, weights, expected_conv, expected_state, mtp=True)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        touched = torch.zeros(state.shape[0], dtype=torch.bool)
        for row, length in enumerate(lengths):
            touched[ids[row, :length].long()] = True
        torch.testing.assert_close(state.cpu()[~touched], before_state.cpu()[~touched], rtol=0, atol=0)
        touched.zero_()
        touched[ids[torch.tensor(lengths) > 0, 0].long()] = True
        torch.testing.assert_close(conv.cpu()[~touched], before_conv.cpu()[~touched], rtol=0, atol=0)
        assert torch.count_nonzero(captured[sum(lengths) :]).item() == 0


def test_first_capture_does_not_publish_weights_to_another_graph():
    # Warm the kernels on another layer, keeping the tested layer's cache empty.
    warm = _fused_case(4)
    for _ in range(3):
        fused_decode_kv(*warm, mtp=True)
    torch.npu.synchronize()
    args = _fused_case(4, strided=True)
    layer, ops, qkv, b, a, z, meta, weights, conv, state = args
    assert not hasattr(layer, "_xllm_kv_conv_weights")
    first = torch.npu.NPUGraph()
    with torch.npu.graph(first, stream=torch.npu.Stream()):
        fused_decode_kv(*args, mtp=True)
    torch.npu.synchronize()
    assert not hasattr(layer, "_xllm_kv_conv_weights")
    second = torch.npu.NPUGraph()
    with torch.npu.graph(second, stream=torch.npu.Stream()):
        captured = fused_decode_kv(*args, mtp=True)
    torch.npu.synchronize()
    assert not hasattr(layer, "_xllm_kv_conv_weights")
    expected_conv, expected_state = conv.clone(), state.clone()
    expected_layer = SimpleNamespace(A_log=layer.A_log, dt_bias=layer.dt_bias, norm=layer.norm)
    expected = fused_decode_kv(
        expected_layer, ops, qkv, b, a, z, meta, weights, expected_conv, expected_state, mtp=True
    )
    # The first graph is deliberately never replayed before the second.
    second.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
