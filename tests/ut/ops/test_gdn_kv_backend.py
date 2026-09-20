# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops import gdn
from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)


def make_metadata(batch=4, sequence=4, live=None):
    live = batch if live is None else live
    spec = sequence > 1
    indices = torch.arange(1, batch * sequence + 1, dtype=torch.int32).reshape(batch, sequence)
    indices[live:] = 0
    starts = torch.arange(batch + 1, dtype=torch.int32).clamp_max(live) * sequence
    meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0 if spec else live,
        num_decode_tokens=0 if spec else live,
        num_spec_decodes=live if spec else 0,
        num_spec_decode_tokens=live * sequence if spec else 0,
        num_actual_tokens=live * sequence,
        spec_sequence_masks=torch.arange(batch) < live if spec else None,
        spec_state_indices_tensor=indices if spec else None,
        non_spec_state_indices_tensor=indices[:, 0] if not spec else None,
    )
    if spec:
        meta.spec_decode_metadata = GDNSpecDecodeMetadata(
            GDNSpecCausalConv1dMetadata(starts, indices, torch.ones(batch, dtype=torch.int32)),
            torch.cat((starts[:1], starts.diff())),
        )
    else:
        meta.non_spec_decode_metadata = GDNDecodeMetadata(
            GDNCausalConv1dMetadata(starts, indices[:, 0], None),
            torch.cat((starts[:1], starts.diff())),
        )
    return meta


@pytest.mark.parametrize("sequence", [1, 4, 17])
@pytest.mark.parametrize("live", [1, 2, 4])
def test_kv_routes_uniform_b4_and_padding(sequence, live):
    metadata = make_metadata(sequence=sequence, live=live)
    choose = gdn._use_xllm_decode if sequence == 1 else gdn._use_xllm_mtp_decode
    with (
        patch.object(gdn, "_GDN_BACKEND", "xllm_ops_kv"),
        patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        assert choose({"layer": metadata}, "layer", False)
        assert not choose({"layer": metadata}, "layer", True)
        metadata.num_prefills = 1
        assert not choose({"layer": metadata}, "layer", False)


def test_ragged_mtp_selects_device_guarded_kv_route():
    metadata = make_metadata()
    metadata.num_actual_tokens -= 1
    metadata.num_spec_decode_tokens -= 1
    with (
        patch.object(gdn, "_GDN_BACKEND", "xllm_ops_kv"),
        patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        assert gdn._use_xllm_mtp_decode({"layer": metadata}, "layer", False)


@pytest.mark.parametrize("backend", ["baseline", "xllm_ops", "xllm_ops_prefill_decode", "xllm_ops_kv"])
def test_non_square_state_shape_matches_backend(backend):
    with patch.object(gdn, "_GDN_BACKEND", backend):
        conv, state = gdn._gdn_state_shape(
            gdn.MambaStateShapeCalculator,
            1,
            1,
            2,
            3,
            5,
            4,
            3,
        )
    assert conv == (6, 16)
    assert state == ((2, 3, 5) if backend == "xllm_ops_kv" else (2, 5, 3))


def test_existing_vk_backend_still_excludes_b4():
    meta = make_metadata()
    with (
        patch.object(gdn, "_GDN_BACKEND", "xllm_ops_prefill_decode"),
        patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        assert not gdn._use_xllm_mtp_decode({"layer": meta}, "layer", False)


@pytest.mark.parametrize("backend", ["xllm_ops_kv", "xllm_ops"])
def test_prefill_minimal_kv_history_and_legacy_checkpoint_contract(backend):
    ids = torch.tensor([4, 1])
    conv = torch.arange(5 * 6 * 6, dtype=torch.float32).reshape(5, 6, 6)
    state = torch.arange(5 * 2 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2, 2)
    before = state.clone()
    conv_before = conv.clone()
    stride = 1 if backend == "xllm_ops_kv" else 4
    metadata = SimpleNamespace(
        prefill_state_indices=ids,
        prefill_has_initial_state=torch.tensor([True, False]),
        prefill_query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        non_spec_prefill_metadata=SimpleNamespace(chunk=SimpleNamespace(cu_seqlens_host=(0, 2, 4))),
    )
    layer = SimpleNamespace(
        num_v_heads=2,
        tp_size=1,
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(2),
        norm=SimpleNamespace(weight=torch.ones(2)),
        _get_xllm_masks=gdn.AscendGatedDeltaNetAttention._get_xllm_masks,
    )

    def prefill(*args, **kwargs):
        if backend == "xllm_ops_kv":
            assert kwargs == {}
            # The native API must receive the original banks and real slot IDs.
            # This catches accidental reintroduction of gather/scatter scratch.
            assert args[5] is conv and args[12] is state
            assert args[8].tolist() == [4, -1]
            assert args[9].tolist() == [4, 1]
            assert args[10].tolist() == [4, -1]
            assert args[11].tolist() == [4, 1]
            assert args[-1] == 4
            conv[ids, :3] = 99
            state[ids] = before[ids] + 10
            return torch.zeros(4, 2, 2)
        assert kwargs == ({} if backend == "xllm_ops_kv" else {"fla_ssm_state_layout": False})
        scratch = args[12]
        assert args[5].shape == (2, 3 if stride == 1 else 6, 6)
        torch.testing.assert_close(args[5], conv_before[ids, : args[5].shape[1]])
        assert scratch.shape == (2 * stride, 2, 2, 2)
        torch.testing.assert_close(scratch[::stride], before[ids])
        assert args[10].tolist() == [0, -1]
        assert args[11].tolist() == [0, stride]
        args[5].fill_(99)
        scratch.fill_(777)
        scratch[::stride] = before[ids] + 10
        return torch.zeros(4, 2, 2)

    with (
        patch.object(gdn, "_GDN_BACKEND", backend),
        patch.dict(sys.modules, {"native_ops_lib": SimpleNamespace(mega_gdn_native_prefill=prefill)}),
        patch.object(
            gdn, "_xllm_ops", SimpleNamespace(mega_gdn_prefill_op=prefill, mega_gdn_prefill_supports_vk_ssm=True)
        ),
    ):
        gdn.AscendGatedDeltaNetAttention._xllm_prefill(
            layer,
            torch.zeros(4, 6),
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            torch.zeros(4, 2, 2),
            metadata,
            torch.zeros(6, 4),
            conv,
            state,
        )
    expected = before.clone()
    expected[ids] += 10
    torch.testing.assert_close(state, expected, rtol=0, atol=0)
    conv_expected = conv_before.clone()
    conv_expected[ids, : 3 if stride == 1 else 6] = 99
    torch.testing.assert_close(conv, conv_expected, rtol=0, atol=0)


@pytest.mark.parametrize("sequence", [1, 4])
def test_kv_fallback_never_calls_vk_recurrent(sequence):
    metadata = make_metadata(sequence=sequence)
    tokens = metadata.num_actual_tokens
    prefix = "layer"

    def arrange(x):
        if x is None:
            return None, None, None
        return tuple(part.reshape(1, -1, 1, 128) for part in x.split(128, dim=-1))

    layer = SimpleNamespace(
        prefix=prefix,
        num_v_heads=1,
        tp_size=1,
        head_v_dim=128,
        head_k_dim=128,
        conv1d=SimpleNamespace(bias=torch.zeros(384), weight=torch.zeros(384, 1, 4)),
        kv_cache=(torch.zeros(64, sequence + 2, 384), torch.zeros(64, 1, 128, 128)),
        activation="silu",
        rearrange_mixed_qkv=arrange,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        norm=Mock(return_value=torch.ones(tokens, 128)),
    )
    fused = Mock(return_value=torch.ones(tokens, 1, 128))

    def convolution(output, x, weights, **kwargs):
        output.copy_(x)

    with (
        patch.object(gdn, "_GDN_BACKEND", "xllm_ops_kv"),
        patch.object(gdn, "get_forward_context", return_value=SimpleNamespace(attn_metadata={prefix: metadata})),
        patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(gdn, "wait_for_kv_layer_from_connector"),
        patch.object(gdn, "record_attention_compute_start"),
        patch.object(gdn, "maybe_save_kv_layer_to_connector"),
        patch.object(gdn, "l2norm_fwd", side_effect=lambda x: x),
        patch.object(
            gdn.DeviceOperator, "fused_gdn_gating", return_value=(torch.zeros(1, tokens, 1), torch.ones(1, tokens, 1))
        ),
        patch.object(gdn, "recurrent_gated_delta_rule_kv", fused),
        patch.object(torch.ops._C_ascend, "npu_causal_conv1d_custom", side_effect=convolution, create=True),
        patch.object(
            torch.ops._C_ascend,
            "npu_recurrent_gated_delta_rule",
            side_effect=AssertionError("VK fallback called"),
            create=True,
        ),
    ):
        output = torch.zeros(tokens, 1, 128)
        gdn.AscendGatedDeltaNetAttention._forward_core(
            layer,
            torch.zeros(tokens, 384),
            torch.zeros(tokens, 1),
            torch.zeros(tokens, 1),
            output,
            torch.zeros_like(output),
        )
    fused.assert_called_once()
    torch.testing.assert_close(output, torch.ones_like(output))


def test_kv_model_allocator_and_layer_queries_agree_for_non_square_state():
    with (
        patch.object(gdn, "_GDN_BACKEND", "xllm_ops_kv"),
        patch.object(gdn.MambaStateShapeCalculator, "gated_delta_net_state_shape", classmethod(gdn._gdn_state_shape)),
    ):
        shape = gdn.MambaStateShapeCalculator.gated_delta_net_state_shape(1, 1, 2, 3, 5, 4, 3)
        layer = SimpleNamespace(
            tp_size=1, num_k_heads=1, num_v_heads=2, head_k_dim=3, head_v_dim=5, conv_kernel_size=4, num_spec=3
        )
        assert gdn.AscendGatedDeltaNetAttention.get_state_shape(layer) == shape
        assert shape[1] == (2, 3, 5)


def test_eager_conv_weight_cache_is_initialized_and_reused_without_npu_probe():
    layer = SimpleNamespace()
    weights = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    with patch.object(torch.npu, "is_current_stream_capturing", side_effect=AssertionError("CPU cache probe")):
        packed = gdn.get_xllm_kv_conv_weights(layer, weights)
        again = gdn.get_xllm_kv_conv_weights(layer, weights)
    torch.testing.assert_close(packed, weights.T)
    assert packed.is_contiguous() and packed.data_ptr() == again.data_ptr()


def _mixed_metadata():
    meta = make_metadata(batch=2, sequence=4)
    meta.num_prefills = 1
    meta.num_prefill_tokens = 129
    meta.num_actual_tokens += 129
    meta.spec_token_indx = torch.arange(8, dtype=torch.int32)
    meta.non_spec_token_indx = torch.arange(8, 137, dtype=torch.int32)
    meta.prefill_state_indices = torch.tensor([20], dtype=torch.int32)
    meta.prefill_query_start_loc = torch.tensor([0, 129], dtype=torch.int32)
    meta.prefill_has_initial_state = torch.tensor([True])
    meta.non_spec_prefill_metadata = SimpleNamespace(chunk=SimpleNamespace(cu_seqlens_host=(0, 129)))
    return meta


@pytest.mark.parametrize("backend", ["baseline", "xllm_ops", "xllm_ops_prefill_decode", "xllm_ops_kv"])
def test_mixed_route_is_native_kv_only(backend):
    meta = _mixed_metadata()
    with (
        patch.object(gdn, "_GDN_BACKEND", backend),
        patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        assert gdn._use_xllm_mixed({"layer": meta}, "layer", False) == (backend == "xllm_ops_kv")
        assert not gdn._use_xllm_mixed({"layer": meta}, "layer", True)
        assert not gdn._use_xllm_mixed(None, "layer", False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_actual_tokens", 136),
        ("num_prefills", 0),
        ("num_spec_decode_tokens", 10),
        ("spec_sequence_masks", None),
        ("spec_token_indx", None),
        ("non_spec_token_indx", None),
        ("spec_decode_metadata", None),
        ("spec_state_indices_tensor", None),
        ("prefill_state_indices", None),
        ("prefill_query_start_loc", None),
        ("prefill_has_initial_state", None),
        ("non_spec_prefill_metadata", None),
    ],
)
def test_mixed_route_rejects_incomplete_or_inconsistent_metadata(field, value):
    meta = _mixed_metadata()
    setattr(meta, field, value)
    with (
        patch.object(gdn, "_GDN_BACKEND", "xllm_ops_kv"),
        patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        assert not gdn._use_xllm_mixed({"layer": meta}, "layer", False)


def test_mixed_route_rejects_pcp():
    with (
        patch.object(gdn, "_GDN_BACKEND", "xllm_ops_kv"),
        patch.object(gdn, "get_pcp_group", return_value=SimpleNamespace(world_size=2)),
    ):
        assert not gdn._use_xllm_mixed({"layer": _mixed_metadata()}, "layer", False)
