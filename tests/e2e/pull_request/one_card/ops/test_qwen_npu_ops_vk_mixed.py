# SPDX-License-Identifier: Apache-2.0
"""Mixed VK routing against independent CPU recurrence, including untouched pages."""

from types import SimpleNamespace as NS

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend import envs
from vllm_ascend.ops.qwen_npu_ops.native_recurrent import NativeVkRecurrent
from vllm_ascend.ops.qwen_npu_ops.prefill import GdnPrefill
from vllm_ascend.ops.qwen_npu_ops.vk_chain import VkGdnChain
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class Norm:
    eps, norm_before_gate = 1e-6, True

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, z):
        value = x.float()
        value = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps) * self.weight).bfloat16()
        return (value.float() * torch.nn.functional.silu(z.float())).bfloat16()


def cpu_sequence(values, history, state):
    outputs = []
    for x, z, b, a in zip(*(values[k] for k in ("x", "z", "b", "a"))):
        window = torch.cat((history, x[None].float()))
        conv = torch.nn.functional.silu((window * values["weight"].float()).sum(0)).bfloat16()
        history = window[1:]
        q, k, v = conv.split((512, 512, 1536))
        q, k = [
            (t.view(4, 128).float() * torch.rsqrt(t.view(4, 128).float().square().sum(-1, keepdim=True) + 1e-6))
            .bfloat16()
            .float()
            .repeat_interleave(3, 0)
            for t in (q, k)
        ]
        g = -values["alog"].exp() * torch.nn.functional.softplus(a.float() + values["bias"])
        beta = b.float().sigmoid().bfloat16().float()
        state = state * g.exp()[:, None, None]
        delta = (v.view(12, 128).float() - (state * k[:, None, :]).sum(-1)) * beta[:, None]
        state += delta[:, :, None] * k[:, None, :]
        core = (state * (q * 128**-0.5)[:, None, :]).sum(-1).bfloat16()
        outputs.append(Norm(values["norm"])(core, z))
    return torch.stack(outputs), history, state


@pytest.fixture(scope="module")
def prefill():
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    init_device_properties_triton()
    return GdnPrefill(envs.VLLM_ASCEND_QWEN_PREFILL_IDENTITY, envs.VLLM_ASCEND_QWEN_PREFILL_BRIDGE, ssm_layout="VK")


@pytest.mark.parametrize("spec,ordinary", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("lengths", [(1, 17), (63, 65), (127, 129)])
@pytest.mark.parametrize(
    "query,ragged",
    [(4, False), (4, True), (4, "zero"), (8, False), (8, True), (9, False), (9, True), (8, "zero"), (9, "zero")],
)
def test_vk_mixed_ragged_prefill_decode(prefill, spec, ordinary, lengths, query, ragged):
    torch.manual_seed(20261108)
    payload = envs.VLLM_ASCEND_QWEN_DECODE_PAYLOAD
    if (query in (4, 9) or ragged) and not payload:
        pytest.skip("Q4/Q9/ragged decode requires the native VK recurrence payload")
    slots = 2 * query + 8
    ordinary_page, prefill_page = 2 * query + 1, 2 * query + 2
    spec_lengths = (
        ((0, query) if ragged == "zero" else (1, min(6, query - 1)) if ragged else (query, query)) if spec else ()
    )
    spec_tokens, ndecode = sum(spec_lengths), int(ordinary)
    count = spec_tokens + ndecode + sum(lengths)
    values = dict(
        x=(torch.randn(count, 2560) / 8).bfloat16(),
        z=(torch.randn(count, 12, 128) / 8).bfloat16(),
        b=(torch.randn(count, 12) / 8).bfloat16(),
        a=(torch.randn(count, 12) / 8 - 1).bfloat16(),
        weight=(torch.randn(4, 2560) / 8).bfloat16(),
        alog=torch.full((12,), -1.0),
        bias=torch.zeros(12),
        norm=torch.ones(128).bfloat16(),
    )
    conv0 = (torch.randn(slots, query + 2, 2560) / 8).bfloat16()
    state0 = torch.randn(slots, 12, 128, 128) / 128
    conv_ref, state_ref = conv0.clone(), state0.clone()
    ids = torch.arange(1, 2 * query + 1, dtype=torch.int32).view(2, query)
    accepted = torch.tensor([query, 2], dtype=torch.int32)
    expected, cursor, modified = [], 0, set()

    def run(length, history, state):
        nonlocal cursor
        piece = {k: v[cursor : cursor + length] if k in ("x", "z", "a", "b") else v for k, v in values.items()}
        emitted, history, state = cpu_sequence(piece, history, state)
        cursor += length
        return emitted, history, state

    for seq, length in enumerate(spec_lengths):
        if not length:
            continue
        first, prev = int(ids[seq, 0]), int(accepted[seq])
        current = state0[int(ids[seq, prev - 1])].clone()
        history = conv0[first, prev - 1 : prev + 2].float()
        start = cursor
        for token in range(length):
            emitted, history, current = run(1, history, current)
            expected.append(emitted)
            state_ref[int(ids[seq, token])] = current
            modified.add(int(ids[seq, token]))
        conv_ref[first, :2] = conv0[first, prev : prev + 2]
        conv_ref[first, 2 : 2 + length] = values["x"][start:cursor]
    if ordinary:
        emitted, _, current = run(1, conv0[ordinary_page, :3].float(), state0[ordinary_page].clone())
        expected.append(emitted)
        conv_ref[ordinary_page, :2] = conv0[ordinary_page, 1:3]
        conv_ref[ordinary_page, 2] = values["x"][spec_tokens]
        state_ref[ordinary_page] = current
        modified.add(ordinary_page)
    for seq, length in enumerate(lengths):
        page = prefill_page + seq
        emitted, history, current = run(
            length,
            conv0[page, :3].float() if seq else torch.zeros(3, 2560),
            state0[page].clone() if seq else torch.zeros(12, 128, 128),
        )
        expected.append(emitted)
        conv_ref[page, :3], state_ref[page] = history.bfloat16(), current
        modified.add(page)
    # Scramble packed rows so dispatch must retain the metadata's group mapping.
    permutation = torch.randperm(count) if spec else torch.arange(count)
    inverse = torch.argsort(permutation)
    device = {k: v.to("npu") for k, v in values.items()}
    inputs = [device[k][permutation.to("npu")] for k in ("x", "z", "b", "a")]
    conv = torch.empty_strided(conv0.shape, ((query + 2) * 2560 + 128, 2560, 1), dtype=torch.bfloat16, device="npu")
    state = torch.empty_strided(state0.shape, (196608 + 64, 16384, 128, 1), dtype=torch.float32, device="npu")
    conv.copy_(conv0)
    state.copy_(state0)
    conv_module = torch.nn.Module()
    conv_module.register_parameter("ascend_conv1d_weight", torch.nn.Parameter(device["weight"], requires_grad=False))
    conv_module.bias = None

    def rearrange(x):
        return tuple(
            t.reshape(1, -1, heads, 128).contiguous() for t, heads in zip(x.split((512, 512, 1536), -1), (4, 4, 12))
        )

    layer = NS(
        num_spec=query - 1,
        head_v_dim=128,
        kv_cache=(conv, state),
        A_log=device["alog"],
        dt_bias=device["bias"],
        norm=Norm(device["norm"]),
        conv1d=conv_module,
        rearrange_mixed_qkv=rearrange,
        activation="silu",
    )
    if payload:
        layer.qwen_recurrent = NativeVkRecurrent(payload)

    def tensor(data):
        return torch.tensor(data, dtype=torch.int32, device="npu")

    meta = NS(
        num_actual_tokens=count,
        num_prefills=2,
        num_decodes=ndecode,
        num_decode_tokens=ndecode,
        num_spec_decodes=2 if spec else 0,
        spec_sequence_masks=object() if spec else None,
        spec_token_indx=inverse[:spec_tokens].to("npu"),
        non_spec_token_indx=inverse[spec_tokens:].to("npu"),
        spec_state_indices_tensor=ids.to("npu"),
        non_spec_state_indices_tensor=tensor(
            [ordinary_page, prefill_page, prefill_page + 1] if ordinary else [prefill_page, prefill_page + 1]
        ),
        spec_decode_metadata=NS(
            spec_causal_conv1d=NS(
                query_start_loc=tensor([0, *torch.tensor(spec_lengths).cumsum(0).tolist()]),
                cache_indices=ids.to("npu"),
                num_accepted_tokens=accepted.to("npu"),
            ),
            actual_seq_lengths=tensor([0, *spec_lengths]),
        ),
        non_spec_decode_metadata=NS(
            causal_conv1d=NS(query_start_loc=tensor([0, 1]), cache_indices=tensor([ordinary_page])),
            actual_seq_lengths=tensor([0, 1]),
        ),
        non_spec_prefill_metadata=NS(chunk=NS(cu_seqlens_host=(0, lengths[0], sum(lengths)))),
        prefill_state_indices=tensor([prefill_page, prefill_page + 1]),
        prefill_has_initial_state=torch.tensor([False, True], device="npu"),
    )
    chain = VkGdnChain(prefill)
    assert chain.supports(layer, meta, *inputs)
    output = torch.full((count + 3, 12, 128), 19, dtype=torch.bfloat16, device="npu")
    assert chain(layer, meta, *inputs, output) == 2
    torch.npu.synchronize()
    expected = torch.cat(expected)[permutation]
    for actual, reference in ((output[:count].cpu(), expected), (state.cpu(), state_ref)):
        delta = actual.float() - reference.float()
        relative = delta.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-12)
        maximum = delta.abs().max() / reference.float().abs().max().clamp_min(1e-12)
        assert torch.isfinite(actual).all() and relative <= 0.02 and maximum <= 0.05, (relative, maximum)
    torch.testing.assert_close(conv.cpu(), conv_ref, rtol=0, atol=0)
    outside = [page for page in range(slots) if page not in modified]
    torch.testing.assert_close(state.cpu()[outside], state0[outside], rtol=0, atol=0)
    assert (output[count:] == 19).all()
