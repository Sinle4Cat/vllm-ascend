# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionMetadataBuilder,
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)


def _decode_metadata(slot=0):
    indices = torch.tensor([slot], dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        non_spec_state_indices_tensor=indices,
    )
    metadata.non_spec_decode_metadata = GDNDecodeMetadata(
        causal_conv1d=GDNCausalConv1dMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            cache_indices=indices,
            initial_state_mode=None,
        ),
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32),
    )
    return metadata


def _mtp_metadata(slots=(9, 4, 11, 2), accepted=2):
    length = len(slots)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=length,
        num_actual_tokens=length,
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        spec_state_indices_tensor=torch.tensor([slots], dtype=torch.int32),
    )
    metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
            query_start_loc=torch.tensor([0, length], dtype=torch.int32),
            cache_indices=metadata.spec_state_indices_tensor,
            num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32),
        ),
        actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32),
    )
    return metadata


def test_mtp_decode_is_opt_in_and_excludes_mixed_partial_or_padded_graphs():
    prefix = "layers.0.linear_attn"
    metadata = _mtp_metadata()
    with patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)):
        for backend in ("baseline", "xllm_ops", "ascendc_phase6"):
            with patch.object(gdn_module, "_GDN_BACKEND", backend):
                assert not gdn_module._use_xllm_mtp_decode({prefix: metadata}, prefix, False)
        with patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops_prefill_decode"):
            assert gdn_module._use_xllm_mtp_decode({prefix: metadata}, prefix, False)
            assert not gdn_module._use_xllm_mtp_decode({prefix: metadata}, prefix, True)
            changes = (
                ("num_prefills", 1),
                ("num_decodes", 1),
                ("num_spec_decodes", 2),
                ("num_actual_tokens", 3),
                ("num_spec_decode_tokens", 3),
                ("spec_sequence_masks", None),
                ("spec_decode_metadata", None),
                ("spec_state_indices_tensor", torch.ones(2, 4, dtype=torch.int32)),
            )
            for name, value in changes:
                old = getattr(metadata, name)
                setattr(metadata, name, value)
                assert not gdn_module._use_xllm_mtp_decode({prefix: metadata}, prefix, False), name
                setattr(metadata, name, old)
    with (
        patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops_prefill_decode"),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=2)),
    ):
        assert not gdn_module._use_xllm_mtp_decode({prefix: metadata}, prefix, False)


def test_mtp_decode_compacts_nonadjacent_checkpoints_and_preserves_vk_layout():
    metadata = _mtp_metadata()
    indices = metadata.spec_state_indices_tensor.flatten().to(torch.long)
    conv = torch.arange(17 * 6 * 384, dtype=torch.float32).reshape(17, 6, 384).to(torch.bfloat16)
    ssm = torch.arange(17 * 128 * 128, dtype=torch.float32).reshape(17, 1, 128, 128)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    layer = SimpleNamespace(
        prefix="layers.0.linear_attn",
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16)),
    )
    packed_pointers = []

    def fake_op(qkv, z, b, a, weights, conv_work, a_log, dt_bias, ssm_work, read, write, accepted, norm, fla):
        assert fla is False and qkv.shape == (1, 4, 384)
        assert conv_work.shape == (1, 6, 384) and ssm_work.shape == (4, 1, 128, 128)
        assert read.tolist() == write.tolist() == [0] and accepted.tolist() in ([1], [4])
        torch.testing.assert_close(ssm_work, ssm.index_select(0, indices))
        packed_pointers.append(weights.data_ptr())
        conv_work.zero_()
        ssm_work.add_(1)
        return qkv, conv_work, ssm_work, torch.ones_like(z)

    def fake_copy(native_conv, native_ssm, conv_work, ssm_work, output, native_indices, query, accepted):
        native_conv.index_copy_(0, native_indices[:1].to(torch.long), conv_work)
        native_ssm.index_copy_(0, native_indices.to(torch.long), ssm_work)

    with (
        patch.object(gdn_module, "_xllm_ops", SimpleNamespace(mega_gdn_mtp_decode=fake_op)),
        patch.object(gdn_module, "copy_mtp_states_to_native", side_effect=fake_copy),
    ):
        for accepted in (1, 4):
            metadata.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens.fill_(accepted)
            out = AscendGatedDeltaNetAttention._xllm_mtp_decode(
                layer,
                torch.ones(4, 384, dtype=torch.bfloat16),
                torch.zeros(4, 1, dtype=torch.bfloat16),
                torch.zeros(4, 1, dtype=torch.bfloat16),
                torch.zeros(4, 1, 128, dtype=torch.bfloat16),
                metadata,
                torch.zeros(384, 4, dtype=torch.bfloat16),
                conv,
                ssm,
            )
            assert out.shape == (4, 1, 128)
    assert packed_pointers[0] == packed_pointers[1]
    selected = torch.zeros(17, dtype=torch.bool)
    selected[indices] = True
    torch.testing.assert_close(ssm[selected], before_ssm[selected] + 2, rtol=0, atol=0)
    torch.testing.assert_close(ssm[~selected], before_ssm[~selected], rtol=0, atol=0)
    selected.zero_()
    selected[indices[0]] = True
    torch.testing.assert_close(conv[~selected], before_conv[~selected], rtol=0, atol=0)
    assert torch.count_nonzero(conv[selected]) == 0


def test_mtp_core_uses_fused_normalization_without_native_chain():
    prefix = "layers.0.linear_attn"
    metadata = _mtp_metadata()
    output = torch.empty(4, 1, 128, dtype=torch.bfloat16)
    fused = Mock(return_value=torch.ones_like(output))
    layer = SimpleNamespace(
        prefix=prefix,
        conv1d=SimpleNamespace(bias=None, weight=torch.zeros(384, 1, 4, dtype=torch.bfloat16)),
        kv_cache=(torch.zeros(17, 6, 384, dtype=torch.bfloat16), torch.zeros(17, 1, 128, 128)),
        num_v_heads=1,
        tp_size=1,
        head_v_dim=128,
        activation="silu",
        norm=SimpleNamespace(
            weight=torch.ones(128, dtype=torch.bfloat16),
            eps=1e-6,
            norm_before_gate=True,
            group_size=None,
            activation="silu",
        ),
        _xllm_mtp_decode=fused,
    )
    with (
        patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops_prefill_decode"),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch(
            "vllm_ascend.ops.gdn.get_forward_context", return_value=SimpleNamespace(attn_metadata={prefix: metadata})
        ),
        patch("vllm_ascend.ops.gdn.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.ops.gdn.record_attention_compute_start"),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer,
            torch.zeros(4, 384, dtype=torch.bfloat16),
            torch.zeros(4, 1, dtype=torch.bfloat16),
            torch.zeros(4, 1, dtype=torch.bfloat16),
            output,
            torch.zeros_like(output),
        )
    fused.assert_called_once()
    torch.testing.assert_close(output, torch.ones_like(output))


def test_real_qwen_patch_binds_all_xllm_helpers():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    from vllm_ascend.patch.worker import patch_qwen3_5

    assert patch_qwen3_5._GDN_PATCH_TARGET is QwenGatedDeltaNetAttention
    for name in ("_xllm_prefill", "_xllm_decode", "_xllm_mtp_decode"):
        assert getattr(QwenGatedDeltaNetAttention, name) is getattr(AscendGatedDeltaNetAttention, name)


@pytest.mark.parametrize("valid_slot", [1023, 1800])
def test_graph_padding_shares_slot_zero_without_remapping_valid_slots(valid_slot):
    builder = SimpleNamespace(
        non_spec_state_indices_tensor=torch.full((8,), 77, dtype=torch.int32),
        non_spec_query_start_loc=torch.full((9,), 77, dtype=torch.int32),
    )
    state_pointer = builder.non_spec_state_indices_tensor.data_ptr()
    query_pointer = builder.non_spec_query_start_loc.data_ptr()
    for slots in ([valid_slot, 7], [valid_slot]):
        count = len(slots)
        padded_states, padded_queries = AscendGDNAttentionMetadataBuilder._pad_non_spec_decode_graph_inputs(
            builder,
            torch.tensor(slots, dtype=torch.int32),
            torch.arange(count + 1, dtype=torch.int32),
            num_decode_tokens=count,
            graph_batch_size=8,
        )
        assert padded_states.data_ptr() == state_pointer
        assert padded_queries.data_ptr() == query_pointer
        assert padded_states.tolist() == slots + [0] * (8 - count)
        assert padded_queries.tolist() == list(range(count + 1)) + [count] * (8 - count)


def test_decode_is_opt_in_and_excludes_padded_or_mixed_graphs():
    prefix = "layers.0.linear_attn"
    metadata = _decode_metadata()
    layer_metadata = {prefix: metadata}
    with patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)):
        for backend in ("baseline", "xllm_ops", "ascendc_phase6"):
            with patch.object(gdn_module, "_GDN_BACKEND", backend):
                assert not gdn_module._use_xllm_decode(layer_metadata, prefix, False)
        with patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops_prefill_decode"):
            assert gdn_module._use_xllm_decode(layer_metadata, prefix, False)
            assert not gdn_module._use_xllm_decode(layer_metadata, prefix, True)
            assert not gdn_module._use_xllm_decode(None, prefix, False)
            assert not gdn_module._use_xllm_decode({}, prefix, False)
            changes = (
                ("num_prefills", 1),
                ("num_decodes", 2),
                ("num_decode_tokens", 2),
                ("num_actual_tokens", 2),
                ("num_spec_decodes", 1),
                ("spec_sequence_masks", torch.ones(1, dtype=torch.bool)),
                ("non_spec_state_indices_tensor", torch.tensor([0, 0], dtype=torch.int32)),
                ("non_spec_decode_metadata", None),
            )
            for name, value in changes:
                old = getattr(metadata, name)
                setattr(metadata, name, value)
                assert not gdn_module._use_xllm_decode(layer_metadata, prefix, False), name
                setattr(metadata, name, old)
    with (
        patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops_prefill_decode"),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=2)),
    ):
        assert not gdn_module._use_xllm_decode(layer_metadata, prefix, False)


@pytest.mark.parametrize("cache_slots", [5, 1025])
def test_decode_preserves_vk_layout_updates_selected_slot_and_reuses_weights(cache_slots):
    metadata = _decode_metadata(cache_slots - 1)
    qkv = torch.ones((1, 384), dtype=torch.bfloat16)
    z = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
    gates = torch.zeros((1, 1), dtype=torch.bfloat16)
    conv_weights = torch.zeros((384, 4), dtype=torch.bfloat16)
    conv_state = torch.zeros((cache_slots, 3, 384), dtype=torch.bfloat16)
    ssm_state = torch.zeros((cache_slots, 1, 128, 128), dtype=torch.float32)
    ssm_state[-1, 0, 3, 7] = 11
    layer = SimpleNamespace(
        prefix="layers.0.linear_attn",
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16)),
    )
    packed_pointers = []

    def fake_op(qkv, z, b, a, weights, conv, a_log, dt_bias, ssm, read, write, norm, fla):
        assert fla is False
        assert weights.shape == (4, 384)
        assert conv.shape[0] == (cache_slots if cache_slots <= 1024 else 1)
        index = read.to(torch.long)
        assert ssm[index, 0, 3, 7].item() in (11, 12)
        assert ssm[index, 0, 7, 3].item() in (0, 1)
        packed_pointers.append(weights.data_ptr())
        conv.index_copy_(0, write.to(torch.long), conv.index_select(0, index) + 1)
        ssm.index_copy_(0, write.to(torch.long), ssm.index_select(0, index) + 1)
        return qkv, conv, ssm, torch.ones_like(z)

    with patch.object(gdn_module, "_xllm_ops", SimpleNamespace(mega_gdn_decode=fake_op)):
        for _ in range(2):
            out = AscendGatedDeltaNetAttention._xllm_decode(
                layer, qkv, gates, gates, z, metadata, conv_weights, conv_state, ssm_state
            )
            torch.testing.assert_close(out, torch.ones_like(z))
    assert packed_pointers[0] == packed_pointers[1]
    assert conv_state[-1, 0, 0].item() == 2
    assert ssm_state[-1, 0, 3, 7].item() == 13
    assert ssm_state[-1, 0, 7, 3].item() == 2
    assert torch.count_nonzero(conv_state[:-1]).item() == 0
    assert torch.count_nonzero(ssm_state[:-1]).item() == 0


def test_decode_core_returns_normalized_output_without_native_chain():
    prefix = "layers.0.linear_attn"
    metadata = _decode_metadata()
    output = torch.empty((1, 1, 128), dtype=torch.bfloat16)
    fused = Mock(return_value=torch.ones_like(output))
    layer = SimpleNamespace(
        prefix=prefix,
        conv1d=SimpleNamespace(bias=None, weight=torch.zeros((384, 1, 4), dtype=torch.bfloat16)),
        kv_cache=(
            torch.zeros((2, 3, 384), dtype=torch.bfloat16),
            torch.zeros((2, 1, 128, 128), dtype=torch.float32),
        ),
        num_v_heads=1,
        tp_size=1,
        head_v_dim=128,
        activation="silu",
        norm=SimpleNamespace(
            weight=torch.ones(128, dtype=torch.bfloat16),
            eps=1e-6,
            norm_before_gate=True,
            group_size=None,
            activation="silu",
        ),
        _xllm_decode=fused,
    )
    with (
        patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops_prefill_decode"),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch(
            "vllm_ascend.ops.gdn.get_forward_context", return_value=SimpleNamespace(attn_metadata={prefix: metadata})
        ),
        patch("vllm_ascend.ops.gdn.wait_for_kv_layer_from_connector") as wait,
        patch("vllm_ascend.ops.gdn.record_attention_compute_start"),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector") as save,
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer,
            torch.zeros((1, 384), dtype=torch.bfloat16),
            torch.zeros((1, 1), dtype=torch.bfloat16),
            torch.zeros((1, 1), dtype=torch.bfloat16),
            output,
            torch.zeros_like(output),
        )
    fused.assert_called_once()
    wait.assert_called_once_with(prefix)
    save.assert_called_once_with("", [])
    torch.testing.assert_close(output, torch.ones_like(output))
