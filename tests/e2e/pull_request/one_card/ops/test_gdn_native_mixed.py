# SPDX-License-Identifier: Apache-2.0
"""Mixed native dispatch against independently packed SDK phase calls."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops import gdn


def _bank(shape, dtype, strided):
    width = 1
    for dim in shape[1:]:
        width *= dim
    storage = torch.full((shape[0], width + (128 if strided else 0)), 0.25, dtype=dtype, device="npu")
    view = storage[:, :width].view(shape)
    view.copy_((torch.randn(shape, device="npu") * 0.03).to(dtype))
    return storage, view


@pytest.mark.parametrize("spec,ordinary", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("lengths", [(1, 129), (257, 130)])
def test_mixed_core_matches_sdk_phases_and_preserves_banks(spec, ordinary, strided, lengths):
    import native_ops_lib

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    init_device_properties_triton()
    torch.manual_seed(2919)
    hk, hv, dim, slots = 4, 12, 128, 48
    channels = (2 * hk + hv) * dim
    num_decode = 2 if ordinary else 0
    spec_tokens = 6 if spec else 0
    non_spec_tokens = num_decode + sum(lengths)
    tokens = non_spec_tokens + spec_tokens
    canonical = [
        (torch.randn(tokens, channels, device="npu") * 0.1).bfloat16(),
        torch.randn(tokens, hv, device="npu").bfloat16(),
        torch.randn(tokens, hv, device="npu").bfloat16(),
        (torch.randn(tokens, hv, dim, device="npu") * 0.1).bfloat16(),
    ]
    order = torch.randperm(tokens) if spec else torch.arange(tokens)
    inverse = order.argsort().npu()
    inputs = [x.index_select(0, order.npu()) for x in canonical]
    if strided:
        combined = torch.cat((inputs[0], inputs[3].flatten(1)), dim=1)
        inputs[0], zflat = combined.split((channels, hv * dim), dim=1)
        inputs[3] = zflat.view(tokens, hv, dim)
    weights = (torch.randn(channels, 4, device="npu") * 0.1).bfloat16()
    packed = weights.T.contiguous()
    conv_storage, conv = _bank((slots, 6, channels), torch.bfloat16, strided)
    state_storage, state = _bank((slots, hv, dim, dim), torch.float32, strided)
    expected_conv_storage = conv_storage.clone()
    expected_state_storage = state_storage.clone()
    expected_conv = expected_conv_storage[:, : conv[0].numel()].view_as(conv)
    expected_state = expected_state_storage[:, : state[0].numel()].view_as(state)
    ids = torch.tensor([[5, 9, 3, 11], [18, 22, 24, 27], [0, 0, 0, 0]], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 4, 6, 6], dtype=torch.int32, device="npu")
    accepted = torch.tensor([3, 1, 1], dtype=torch.int32, device="npu")
    prefill_ids = torch.tensor([29, 34], dtype=torch.int32, device="npu")
    prefill_starts = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device="npu")
    initial = torch.tensor([True, False], device="npu")
    ordinary_ids = torch.tensor([41, 44], dtype=torch.int32, device="npu")
    non_spec_ids = torch.cat((ordinary_ids, prefill_ids)) if ordinary else prefill_ids
    non_spec_starts = torch.tensor(
        [0, 1, 2, 2 + lengths[0], 2 + sum(lengths)] if ordinary else [0, lengths[0], sum(lengths)],
        dtype=torch.int32,
        device="npu",
    )
    meta = GDNAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=sum(lengths),
        num_decodes=num_decode,
        num_decode_tokens=num_decode,
        num_spec_decodes=2 if spec else 0,
        num_spec_decode_tokens=spec_tokens,
        num_actual_tokens=tokens,
        spec_sequence_masks=torch.tensor([True, False, True, False], device="npu") if spec else None,
        spec_state_indices_tensor=ids if spec else None,
        non_spec_state_indices_tensor=non_spec_ids,
        spec_token_indx=inverse[non_spec_tokens:] if spec else None,
        non_spec_token_indx=inverse[:non_spec_tokens] if spec else None,
        prefill_state_indices=prefill_ids,
        prefill_query_start_loc=prefill_starts,
        prefill_has_initial_state=initial,
    )
    meta.spec_decode_metadata = (
        SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=starts,
                num_accepted_tokens=accepted,
            )
        )
        if spec
        else None
    )
    meta.non_spec_decode_metadata = SimpleNamespace(causal_conv1d=SimpleNamespace(query_start_loc=non_spec_starts))
    meta.non_spec_prefill_metadata = SimpleNamespace(
        chunk=SimpleNamespace(cu_seqlens_host=(0, lengths[0], sum(lengths)))
    )
    layer = SimpleNamespace(
        prefix="layer",
        num_v_heads=hv,
        tp_size=1,
        head_k_dim=dim,
        head_v_dim=dim,
        A_log=torch.zeros(hv, device="npu"),
        dt_bias=torch.zeros(hv, device="npu"),
        norm=SimpleNamespace(
            weight=torch.ones(dim, dtype=torch.bfloat16, device="npu"),
            eps=1e-6,
            norm_before_gate=True,
            group_size=None,
            activation="silu",
        ),
        conv1d=SimpleNamespace(bias=None, weight=weights.unsqueeze(1)),
        activation="silu",
        kv_cache=(conv, state),
        _get_xllm_masks=gdn.AscendGatedDeltaNetAttention._get_xllm_masks,
    )
    masks = layer._get_xllm_masks(conv.device, conv.dtype)
    for repeat in range(2):
        before_conv, before_state = conv_storage.clone(), state_storage.clone()
        expected = torch.empty_like(canonical[3])
        if spec:
            q, b, a, z = [x[non_spec_tokens:] for x in canonical]
            expected[non_spec_tokens:] = native_ops_lib.mega_gdn_native_decode(
                q,
                z,
                b,
                a,
                packed,
                expected_conv,
                layer.A_log,
                layer.dt_bias,
                expected_state,
                ids,
                starts,
                accepted,
                layer.norm.weight,
            )
        if ordinary:
            q, b, a, z = [x[:2] for x in canonical]
            expected[:2] = native_ops_lib.mega_gdn_native_decode(
                q,
                z,
                b,
                a,
                packed,
                expected_conv,
                layer.A_log,
                layer.dt_bias,
                expected_state,
                ordinary_ids[:, None].contiguous(),
                non_spec_starts[:3],
                torch.ones_like(ordinary_ids),
                layer.norm.weight,
            )
        q, b, a, z = [x[num_decode:non_spec_tokens] for x in canonical]
        read = torch.where(initial, prefill_ids, -1)
        expected[num_decode:non_spec_tokens] = native_ops_lib.mega_gdn_native_prefill(
            q,
            b,
            a,
            z,
            packed,
            expected_conv,
            layer.A_log,
            layer.dt_bias,
            read,
            prefill_ids,
            read,
            prefill_ids,
            expected_state,
            *masks,
            prefill_starts,
            layer.norm.weight,
            sum((n + 127) // 128 for n in lengths) * hv,
        )
        output = torch.full_like(inputs[3], float("nan"))
        with (
            patch.object(gdn, "_GDN_BACKEND", "xllm_ops_kv"),
            patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
            patch.object(gdn, "get_forward_context", return_value=SimpleNamespace(attn_metadata={"layer": meta})),
            patch.object(gdn, "wait_for_kv_layer_from_connector"),
            patch.object(gdn, "record_attention_compute_start"),
            patch.object(gdn, "maybe_save_kv_layer_to_connector"),
            patch.object(gdn, "chunk_gated_delta_rule", side_effect=AssertionError("mixed path fell back")),
        ):
            assert gdn._use_xllm_mixed({"layer": meta}, "layer", False)
            gdn.AscendGatedDeltaNetAttention._forward_core(layer, *inputs[:3], output, inputs[3])
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected.index_select(0, order.npu()), rtol=0, atol=0)
        torch.testing.assert_close(conv_storage, expected_conv_storage, rtol=0, atol=0)
        torch.testing.assert_close(state_storage, expected_state_storage, rtol=0, atol=0)
        touched_state = [29, 34] + ([41, 44] if ordinary else []) + ([5, 9, 3, 11, 18, 22] if spec else [])
        touched_conv = [29, 34] + ([41, 44] if ordinary else []) + ([5, 18] if spec else [])
        for actual, before, touched in [
            (conv_storage, before_conv, touched_conv),
            (state_storage, before_state, touched_state),
        ]:
            inactive = torch.tensor([i for i in range(slots) if i not in touched], device="npu")
            torch.testing.assert_close(
                actual.index_select(0, inactive), before.index_select(0, inactive), rtol=0, atol=0
            )
        torch.testing.assert_close(
            conv[prefill_ids.long(), 3:],
            before_conv[prefill_ids.long(), : conv[0].numel()].view(2, 6, channels)[:, 3:],
            rtol=0,
            atol=0,
        )
        if strided:
            torch.testing.assert_close(
                conv_storage[:, conv[0].numel() :], before_conv[:, conv[0].numel() :], rtol=0, atol=0
            )
            torch.testing.assert_close(
                state_storage[:, state[0].numel() :], before_state[:, state[0].numel() :], rtol=0, atol=0
            )
        initial.fill_(True)
        accepted.copy_(torch.tensor([1, 4, 1], dtype=torch.int32, device="npu"))
