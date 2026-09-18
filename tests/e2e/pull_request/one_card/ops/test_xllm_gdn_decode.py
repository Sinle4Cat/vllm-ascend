# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from types import SimpleNamespace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops_lib")

from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.layernorm import LayerNormFn
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")

HEAD_DIM = 128
KEY_HEADS = 4
VALUE_HEADS = 12
CONV_DIM = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM


def make_case(cache_slots, padded=False):
    generator = torch.Generator().manual_seed(20260917)

    def rand(shape, scale=0.2, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator) * scale).to(dtype).to("npu:0")

    case = SimpleNamespace(
        qkv=rand((1, CONV_DIM)),
        z=rand((1, VALUE_HEADS, HEAD_DIM)),
        b=rand((1, VALUE_HEADS), 0.3),
        a=rand((1, VALUE_HEADS)),
        weights=rand((CONV_DIM, 4), 0.15),
        conv=rand((cache_slots, 3, CONV_DIM)),
        ssm=rand((cache_slots, VALUE_HEADS, HEAD_DIM, HEAD_DIM), 0.02, torch.float32),
        a_log=rand((VALUE_HEADS,), 0.1, torch.float32) - 0.5,
        dt_bias=rand((VALUE_HEADS,), 0.1, torch.float32),
        norm_weight=1 + rand((HEAD_DIM,), 0.05),
        indices=torch.tensor([cache_slots - 1], dtype=torch.int32, device="npu:0"),
        query_start=torch.tensor([0, 1], dtype=torch.int32, device="npu:0"),
    )
    case.layer = SimpleNamespace(
        prefix="test.linear_attn",
        A_log=case.a_log,
        dt_bias=case.dt_bias,
        norm=SimpleNamespace(weight=case.norm_weight),
    )
    case.metadata = SimpleNamespace(non_spec_state_indices_tensor=case.indices)
    if padded:
        # Match block-level padding: logical entries are dense, outer rows are not.
        conv_backing = torch.zeros((cache_slots, 3 * CONV_DIM + 64), dtype=case.conv.dtype, device=case.conv.device)
        ssm_backing = torch.zeros(
            (cache_slots, VALUE_HEADS * HEAD_DIM * HEAD_DIM + 512), dtype=case.ssm.dtype, device=case.ssm.device
        )
        conv_view = conv_backing[:, : 3 * CONV_DIM].view_as(case.conv)
        ssm_view = ssm_backing[:, : VALUE_HEADS * HEAD_DIM * HEAD_DIM].view_as(case.ssm)
        conv_view.copy_(case.conv)
        ssm_view.copy_(case.ssm)
        case.conv, case.ssm = conv_view, ssm_view
        assert not case.conv.is_contiguous() and not case.ssm.is_contiguous()
    return case


def native_decode(case):
    assert case.conv.is_contiguous() and case.ssm.is_contiguous(), "native reference requires dense caches"
    conv_out = torch.empty_like(case.qkv)
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        conv_out,
        case.qkv,
        case.weights.transpose(0, 1).contiguous(),
        conv_state=case.conv,
        bias_opt=None,
        query_start_loc_opt=case.query_start,
        cache_indices_opt=case.indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=None,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    query, key, value = torch.split(
        conv_out, [KEY_HEADS * HEAD_DIM, KEY_HEADS * HEAD_DIM, VALUE_HEADS * HEAD_DIM], dim=-1
    )
    query = l2norm_fwd(query.reshape(1, 1, KEY_HEADS, HEAD_DIM)).squeeze(0)
    key = l2norm_fwd(key.reshape(1, 1, KEY_HEADS, HEAD_DIM)).squeeze(0)
    g, beta = fused_gdn_gating_patch(case.a_log, case.a, case.b, case.dt_bias)
    core = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query,
        key=key,
        value=value.reshape(1, VALUE_HEADS, HEAD_DIM),
        g=g.squeeze(0),
        beta=beta.squeeze(0),
        state=case.ssm,
        scale=HEAD_DIM**-0.5,
        actual_seq_lengths=case.query_start,
        ssm_state_indices=case.indices,
    )
    return LayerNormFn.apply(
        core.reshape(-1, HEAD_DIM),
        case.norm_weight,
        None,
        case.z.reshape(-1, HEAD_DIM),
        1e-6,
        None,
        True,
        True,
    ).reshape(1, VALUE_HEADS, HEAD_DIM)


def fused_decode(case):
    return AscendGatedDeltaNetAttention._xllm_decode(
        case.layer,
        case.qkv,
        case.b,
        case.a,
        case.z,
        case.metadata,
        case.weights,
        case.conv,
        case.ssm,
    )


@pytest.fixture(autouse=True)
def setup_ops(monkeypatch):
    torch.npu.set_device(0)
    init_device_properties_triton()
    assert enable_custom_op(), "vLLM-Ascend custom extension is required"
    monkeypatch.setattr(gdn_module, "_xllm_ops", custom_ops)


@pytest.mark.parametrize("cache_slots,padded", [(8, False), (1025, False), (8, True)])
def test_native_output_and_state_vs_megagdn(cache_slots, padded):
    # Ascend's production Mamba allocator deliberately keeps native states dense.
    # Validate a strided Mega adapter against the equivalent dense native state,
    # not an unsupported raw-strided call into the native reference kernels.
    native, fused = make_case(cache_slots), make_case(cache_slots, padded)
    expected, actual = native_decode(native), fused_decode(fused)
    torch.npu.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.005)
    torch.testing.assert_close(fused.conv, native.conv, rtol=0, atol=0)
    torch.testing.assert_close(fused.ssm, native.ssm, rtol=0.005, atol=0.0005)


@pytest.mark.parametrize("cache_slots,padded", [(8, False), (1025, False), (8, True)])
def test_graph_replay_updates_current_input_and_slot(cache_slots, padded):
    eager, captured = make_case(cache_slots, padded), make_case(cache_slots, padded)
    initial_conv, initial_ssm = captured.conv.clone(), captured.ssm.clone()
    for _ in range(3):
        fused_decode(captured)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = fused_decode(captured)
    captured.conv.copy_(initial_conv)
    captured.ssm.copy_(initial_ssm)
    for step in range(8):
        slot = cache_slots - 1 if step % 2 == 0 else 1
        eager.indices.fill_(slot)
        captured.indices.fill_(slot)
        eager.qkv.add_(0.01)
        captured.qkv.copy_(eager.qkv)
        expected = fused_decode(eager)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(captured.conv, eager.conv, rtol=0, atol=0)
        torch.testing.assert_close(captured.ssm, eager.ssm, rtol=0, atol=0)
