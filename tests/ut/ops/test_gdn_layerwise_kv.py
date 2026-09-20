# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    qwen_gdn_attention_core,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNPrefillMetadata,
)


class _Linear(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states[:, : self.output_size], None


class _Norm(nn.Module):
    def forward(self, hidden_states: torch.Tensor, z: torch.Tensor):
        del z
        return hidden_states


class _OutputProjection(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states, None


class _GDNForwardWrapper(nn.Module):
    forward = AscendGatedDeltaNetAttention.forward
    _forward_core = AscendGatedDeltaNetAttention._forward_core
    _split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = _Linear(2)
        self.in_proj_ba = _Linear(2)
        self.in_proj_z = _Linear(2)
        self.norm = _Norm()
        self.out_proj = _OutputProjection()
        self.conv1d = nn.Conv1d(1, 2, kernel_size=2)
        self.num_v_heads = 1
        self.tp_size = 1
        self.head_v_dim = 2
        self.activation = None
        self.register_buffer("A_log", torch.zeros(1))
        self.register_buffer("dt_bias", torch.zeros(1))
        self.register_buffer("conv_state", torch.zeros(1, 1, 2))
        self.register_buffer("ssm_state", torch.zeros(1, 1, 2, 2))
        self.prefix = "layers.0.linear_attn"

    @property
    def kv_cache(self):
        return self.conv_state, self.ssm_state

    def split_ba(self, ba: torch.Tensor):
        # Avoid torch.chunk: torch-npu's aten.split fallback conflicts with
        # PyTorch's decomposition check when CI is set.
        midpoint = ba.shape[-1] // 2
        return ba[..., :midpoint], ba[..., midpoint:]

    def rearrange_mixed_qkv(self, mixed_qkv: torch.Tensor | None):
        if mixed_qkv is None:
            return None, None, None
        projected = mixed_qkv.reshape(1, mixed_qkv.shape[0], 1, 2)
        return projected, projected, projected


def _run_gdn_forward(
    model: nn.Module,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    return model(hidden_states)


def _make_prefill_metadata(device: torch.device | str = "cpu") -> GDNAttentionMetadata:
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_state_indices_tensor=torch.tensor(
            [0], dtype=torch.int32, device=device
        ),
    )
    # These fields are constructor arguments only in newer vLLM releases.
    # Assigning them after construction also supports the older metadata class
    # while exercising the same production GDN prefill path.
    metadata.prefill_query_start_loc = torch.tensor(
        [0, 2], dtype=torch.int32, device=device
    )
    metadata.prefill_state_indices = torch.tensor([0], dtype=torch.int64, device=device)
    metadata.prefill_has_initial_state = torch.tensor([True], device=device)
    metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
        causal_conv1d=GDNCausalConv1dMetadata(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
            cache_indices=torch.tensor([0], dtype=torch.int32, device=device),
            initial_state_mode=torch.tensor([1], dtype=torch.int32, device=device),
        ),
        chunk=Mock(),
    )
    return metadata


def test_xllm_backend_is_routed_only_to_pure_prefill():
    metadata = _make_prefill_metadata()
    layer_metadata = {"layers.0.linear_attn": metadata}

    with (
        patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops"),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        assert gdn_module._use_xllm_prefill(
            layer_metadata, "layers.0.linear_attn", False
        )

        metadata.num_prefills = 0
        metadata.num_decodes = 1
        assert not gdn_module._use_xllm_prefill(
            layer_metadata, "layers.0.linear_attn", False
        )

        metadata.num_prefills = 1
        assert not gdn_module._use_xllm_prefill(
            layer_metadata, "layers.0.linear_attn", False
        )

        metadata.num_decodes = 0
        metadata.spec_sequence_masks = torch.ones(1, dtype=torch.bool)
        assert not gdn_module._use_xllm_prefill(
            layer_metadata, "layers.0.linear_attn", False
        )

        metadata.spec_sequence_masks = None
        assert not gdn_module._use_xllm_prefill(
            layer_metadata, "layers.0.linear_attn", True
        )

    with (
        patch.object(gdn_module, "_GDN_BACKEND", "xllm_ops"),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=2),
        ),
    ):
        assert not gdn_module._use_xllm_prefill(
            layer_metadata, "layers.0.linear_attn", False
        )


@pytest.mark.parametrize("checkpoint_stride", [1, 2, 5])
@pytest.mark.parametrize("initial_state_flags", [(True, False), (False, True)])
@pytest.mark.parametrize("supports_vk_ssm", [False, True])
def test_xllm_prefill_checkpoint_scratch_preserves_native_cache(
    checkpoint_stride, initial_state_flags, supports_vk_ssm
):
    state_indices = torch.tensor([4, 1], dtype=torch.long)
    has_initial_state = torch.tensor(initial_state_flags)
    conv_state = torch.arange(
        5 * (checkpoint_stride + 2) * 6, dtype=torch.float32
    ).reshape(5, checkpoint_stride + 2, 6)
    ssm_state = torch.arange(5 * 2 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2, 2)
    conv_before, ssm_before = conv_state.clone(), ssm_state.clone()
    final_ssm_kv = (
        torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2) + 100
    )
    expected_output = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2)
    metadata = _make_prefill_metadata()
    metadata.num_prefills = 2
    metadata.num_prefill_tokens = 4
    metadata.num_actual_tokens = 4
    metadata.prefill_state_indices = state_indices
    metadata.prefill_has_initial_state = has_initial_state
    metadata.prefill_query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata.non_spec_prefill_metadata.chunk.cu_seqlens_host = (0, 2, 4)
    model = SimpleNamespace(
        num_v_heads=2,
        tp_size=1,
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(2),
        norm=SimpleNamespace(weight=torch.ones(2)),
        _get_xllm_masks=AscendGatedDeltaNetAttention._get_xllm_masks,
    )

    def fused_prefill(*args, **kwargs):
        assert kwargs == ({"fla_ssm_state_layout": False} if supports_vk_ssm else {})
        conv_scratch, ssm_scratch = args[5], args[12]
        conv_read, conv_write, ssm_read, ssm_write = args[8:12]
        assert conv_scratch.shape == (2, checkpoint_stride + 2, 6)
        assert ssm_scratch.shape == (2 * checkpoint_stride, 2, 2, 2)
        assert conv_scratch.is_contiguous() and ssm_scratch.is_contiguous()
        expected_read = [
            index if initialized else -1
            for index, initialized in enumerate(initial_state_flags)
        ]
        assert conv_read.tolist() == expected_read
        assert conv_write.tolist() == [0, 1]
        assert ssm_read.tolist() == [
            index * checkpoint_stride if index >= 0 else -1 for index in expected_read
        ]
        assert ssm_write.tolist() == [0, checkpoint_stride]
        torch.testing.assert_close(
            ssm_scratch[::checkpoint_stride],
            ssm_before.index_select(0, state_indices)
            if supports_vk_ssm
            else ssm_before.index_select(0, state_indices).transpose(-1, -2),
        )
        # Reserved checkpoint sentinels must never be copied into native state.
        ssm_scratch.fill_(777)
        ssm_scratch.index_copy_(
            0,
            ssm_write.long(),
            final_ssm_kv.transpose(-1, -2).contiguous()
            if supports_vk_ssm
            else final_ssm_kv,
        )
        conv_scratch[:, :3].add_(10)
        return expected_output

    fused_op = Mock(side_effect=fused_prefill)
    with patch.object(
        gdn_module,
        "_xllm_ops",
        SimpleNamespace(
            mega_gdn_prefill_op=fused_op,
            mega_gdn_prefill_supports_vk_ssm=supports_vk_ssm,
        ),
    ):
        output = AscendGatedDeltaNetAttention._xllm_prefill(
            model,
            torch.zeros(4, 6),
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            torch.zeros(4, 2, 2),
            metadata,
            torch.ones(6, 4),
            conv_state,
            ssm_state,
        )
    fused_op.assert_called_once()
    assert output is expected_output
    expected_ssm = ssm_before.clone()
    expected_ssm.index_copy_(
        0, state_indices, final_ssm_kv.transpose(-1, -2).contiguous()
    )
    torch.testing.assert_close(ssm_state, expected_ssm)
    expected_conv = conv_before.clone()
    selected_conv = conv_before.index_select(0, state_indices)
    selected_conv[:, :3].add_(10)
    expected_conv.index_copy_(0, state_indices, selected_conv)
    torch.testing.assert_close(conv_state, expected_conv)


def test_connector_observes_updated_gdn_state_for_each_compiled_call():
    # vLLM registers this production dispatcher for NPU only. Keep a CPU
    # registration alive for this test so Inductor sees the same custom op.
    cpu_impl = torch.library.Library("vllm", "IMPL", "CPU")
    cpu_impl.impl("qwen_gdn_attention_core", qwen_gdn_attention_core)
    model = _GDNForwardWrapper()
    hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    output = torch.empty_like(hidden_states)
    metadata = _make_prefill_metadata()
    forward_context = ForwardContext(
        no_compile_layers={model.prefix: model},
        attn_metadata={model.prefix: metadata},
        slot_mapping={},
    )
    connector = Mock()
    observed_states = []

    def record_ready_state(layer_name, kv_cache_layer, attn_metadata):
        assert layer_name == ""
        assert kv_cache_layer == []
        assert attn_metadata is forward_context.attn_metadata
        observed_states.append(tuple(state.clone() for state in model.kv_cache))

    connector.save_kv_layer.side_effect = record_ready_state

    def causal_conv1d(output_tensor, mixed_qkv, conv_weights, **kwargs):
        del conv_weights, kwargs
        output_tensor.copy_(mixed_qkv)
        model.kv_cache[0].add_(1)

    def chunk_attention(**kwargs):
        initial_state = kwargs["initial_state"]
        value = kwargs["v"]
        return value + 1, initial_state + 1

    gating = (
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 1),
    )

    with (
        override_forward_context(forward_context),
        patch.object(torch.accelerator, "is_available", return_value=False),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating", return_value=gating
        ),
        patch("vllm_ascend.ops.gdn.clear_ssm_states"),
        patch(
            "vllm_ascend.ops.gdn.chunk_gated_delta_rule", side_effect=chunk_attention
        ),
        patch.object(
            AscendGatedDeltaNetAttention, "_probe_fused_chunk", return_value=False
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_causal_conv1d_custom",
            side_effect=causal_conv1d,
            create=True,
        ),
        patch("vllm_ascend.attention.utils.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.attention.utils.is_v1_kv_transfer_group", return_value=True),
        patch(
            "vllm_ascend.attention.utils.get_kv_transfer_group", return_value=connector
        ),
        # NPU-only side effects live in the compiled forward on device; stub them
        # so the CPU inductor graph stays fullgraph (no graph break).
        patch(
            "vllm_ascend.ops.gdn.wait_for_kv_layer_from_connector", lambda *a, **k: None
        ),
        patch("vllm_ascend.ops.gdn.record_attention_compute_start", lambda: None),
    ):
        eager_output = _run_gdn_forward(model, hidden_states, output)
        torch.testing.assert_close(eager_output, hidden_states + 1)
        compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
        for _ in range(2):
            compiled_output = _run_gdn_forward(compiled_model, hidden_states, output)
            torch.testing.assert_close(compiled_output, hidden_states + 1)

    assert connector.save_kv_layer.call_count == 3
    for execution, (conv_state, ssm_state) in enumerate(observed_states, start=1):
        torch.testing.assert_close(conv_state, torch.full_like(conv_state, execution))
        torch.testing.assert_close(ssm_state, torch.full_like(ssm_state, execution))
