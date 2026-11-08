# SPDX-License-Identifier: Apache-2.0
"""Independent VK state checks for compact queries and dynamic Graph metadata."""

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

from vllm_ascend import envs
from vllm_ascend.ops.qwen_npu_ops.native_recurrent import NativeVkRecurrent


@pytest.fixture(scope="module")
def recurrence():
    payload = envs.VLLM_ASCEND_QWEN_DECODE_PAYLOAD
    if not payload:
        pytest.skip("Set VLLM_ASCEND_QWEN_DECODE_PAYLOAD to a compiled native VK payload")
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    return NativeVkRecurrent(payload)


def golden(q, k, v, g, beta, initial, ids, lengths, accepted):
    state, output, cursor = initial.clone(), torch.zeros_like(v), 0
    for batch, length in enumerate(lengths):
        if not length:
            continue
        current = state[ids[batch, accepted[batch] - 1]].clone()
        for token in range(length):
            row = cursor + token
            key = k[row].float().repeat_interleave(3, 0)
            query = q[row].float().repeat_interleave(3, 0) * 128**-0.5
            current *= g[row].exp()[:, None, None]
            delta = (v[row].float() - (current * key[:, None, :]).sum(-1)) * beta[row].float()[:, None]
            current += delta[:, :, None] * key[:, None, :]
            output[row] = (current * query[:, None, :]).sum(-1).bfloat16()
            state[ids[batch, token]] = current
        cursor += length
    return output, state


def relative_error(actual, expected):
    difference = (actual.float() - expected.float()).square().sum().sqrt()
    return difference / expected.float().square().sum().sqrt().clamp_min(1e-12)


def make_inputs(tokens):
    q, k = [torch.nn.functional.normalize(torch.randn(tokens, 4, 128), dim=-1).bfloat16() for _ in range(2)]
    return (
        q,
        k,
        (torch.randn(tokens, 12, 128) / 8).bfloat16(),
        -torch.rand(tokens, 12) / 5,
        torch.rand(tokens, 12).bfloat16(),
    )


def launch(recurrence, inputs, state, lens, ids, accepted):
    q, k, v, g, beta = inputs
    return recurrence(
        q,
        k,
        v,
        state,
        g=g,
        beta=beta,
        scale=128**-0.5,
        actual_seq_lengths=lens,
        ssm_state_indices=ids,
        num_accepted_tokens=accepted,
    )


@pytest.mark.parametrize("width", [8, 9])
@pytest.mark.parametrize(
    "kind", ["full", "ragged", "zero-middle", "zero-first", "batch8-short", "batch8", "batch8-ragged"]
)
def test_native_vk_full_state(recurrence, width, kind):
    torch.manual_seed(20261108)
    lengths = {
        "full": [width, width],
        "ragged": [1, 6],
        "zero-middle": [width, 1, 0, 3],
        "zero-first": [0, 1, width, 0],
        "batch8-short": [2] * 8,
        "batch8": [width] * 8,
        "batch8-ragged": [0, width, 1, width - 1, 2, width - 2, 3, width - 3],
    }[kind]
    _check_case(recurrence, width, lengths, parity=width == 8 and kind in ("full", "batch8"))


@pytest.mark.parametrize("width,lengths,graph", [(1, [1, 0, 1], False), (9, [1, 6], True)])
def test_native_vk_ordinary_and_graph(recurrence, width, lengths, graph):
    torch.manual_seed(20261108)
    _check_case(recurrence, width, lengths, graph=graph)


def _check_case(recurrence, width, lengths, *, graph=False, parity=False):
    batches, tokens = len(lengths), sum(lengths)
    slots = batches * width + 2
    backing = torch.randn(slots, 12 * 128 * 128 + 32) / 128
    initial = backing[:, : 12 * 128 * 128].view(slots, 12, 128, 128)
    device_backing = backing.to("npu")
    state = device_backing[:, : 12 * 128 * 128].view_as(initial)
    inputs = make_inputs(tokens)
    ids = torch.arange(1, slots - 1, dtype=torch.int32).view(batches, width)
    ids[torch.tensor(lengths) == 0] = 0
    accepted = [width if batch % 2 == 0 else min(2, width) for batch in range(batches)]
    expected, expected_state = golden(*inputs, initial, ids, lengths, accepted)
    device_inputs = [value.to("npu") for value in inputs]
    lens = torch.tensor([0, *lengths], dtype=torch.int32, device="npu")
    device_ids = ids.to("npu")
    device_accepted = torch.tensor(accepted, dtype=torch.int32, device="npu")
    if graph:
        launch(recurrence, device_inputs, state, lens, device_ids, device_accepted)
        torch.npu.synchronize()
        state.copy_(initial)
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = launch(recurrence, device_inputs, state, lens, device_ids, device_accepted)
        state.copy_(initial)
        captured.replay()
    else:
        output = launch(recurrence, device_inputs, state, lens, device_ids, device_accepted)
    torch.npu.synchronize()
    actual, actual_state = output.cpu(), state.cpu()
    assert relative_error(actual, expected) < 0.005
    assert relative_error(actual_state, expected_state) < 0.001
    destinations = {int(ids[batch, token]) for batch, length in enumerate(lengths) for token in range(length)}
    untouched = [page for page in range(slots) if page not in destinations]
    torch.testing.assert_close(actual_state[untouched], initial[untouched], rtol=0, atol=0)
    torch.testing.assert_close(device_backing.cpu()[:, -32:], backing[:, -32:], rtol=0, atol=0)
    if parity:
        state.copy_(initial)
        q, k, v, g, beta = device_inputs
        native = recurrent_gated_delta_rule(
            q,
            k,
            v,
            state,
            g=g,
            beta=beta,
            scale=128**-0.5,
            actual_seq_lengths=lens,
            ssm_state_indices=device_ids.flatten(),
            num_accepted_tokens=device_accepted,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(native.cpu(), actual, rtol=0, atol=0)
        torch.testing.assert_close(state.cpu(), actual_state, rtol=0, atol=0)


@pytest.mark.parametrize("width", [8, 9])
@pytest.mark.parametrize("case", ["ragged", "reversed", "all-zero"])
def test_native_vk_graph_metadata_changes(recurrence, width, case):
    torch.manual_seed(20261109)
    batches, tokens = 3, 3 * width
    slots = tokens + 2
    initial = torch.randn(slots, 12, 128, 128) / 128
    state = initial.to("npu")
    inputs = make_inputs(tokens)
    device_inputs = [value.to("npu") for value in inputs]
    table = torch.arange(1, slots - 1, dtype=torch.int32).view(batches, width)
    ids = table.to("npu")
    lens = torch.tensor([0, width, width, width], dtype=torch.int32, device="npu")
    accepted = torch.ones(batches, dtype=torch.int32, device="npu")
    launch(recurrence, device_inputs, state, lens, ids, accepted)
    torch.npu.synchronize()
    captured = torch.npu.NPUGraph()
    with torch.npu.graph(captured):
        output = launch(recurrence, device_inputs, state, lens, ids, accepted)
    lengths, counts = {
        "ragged": ([1, 6, 0], [width, 2, 0]),
        "reversed": ([0, width, 1], [0, width, 1]),
        "all-zero": ([0, 0, 0], [0, 0, 0]),
    }[case]
    pages = table.flip(1).contiguous() if case == "reversed" else table.clone()
    pages[torch.tensor(lengths) == 0] = 0
    state.copy_(initial)
    ids.copy_(pages)
    lens.copy_(torch.tensor([0, *lengths], dtype=torch.int32))
    accepted.copy_(torch.tensor(counts, dtype=torch.int32))
    output.fill_(19)
    captured.replay()
    torch.npu.synchronize()
    expected, expected_state = golden(*inputs, initial, pages, lengths, counts)
    actual, actual_state = output.cpu(), state.cpu()
    count = sum(lengths)
    assert relative_error(actual[:count], expected[:count]) < 0.005
    assert relative_error(actual_state, expected_state) < 0.001
    destinations = {int(pages[batch, token]) for batch, length in enumerate(lengths) for token in range(length)}
    untouched = [page for page in range(slots) if page not in destinations]
    torch.testing.assert_close(actual_state[untouched], initial[untouched], rtol=0, atol=0)
    assert (actual[count:] == 19).all()
