# SPDX-License-Identifier: Apache-2.0
"""NPU state-bridge regression without model weights or a private payload."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.qwen_npu_ops.state_bridge import gather, prepare, scatter
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("layout", ["KV", "VK"])
@pytest.mark.parametrize("query", [4, 8, 9])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_state_bridge_preserves_layout_stride_and_inactive_pages(layout, query, graph_mode):
    device = "npu:0"
    torch.npu.set_device(0)
    init_device_properties_triton()
    slots = 7
    torch.manual_seed(20261108)
    # Slice larger banks to keep a padded outer page stride.
    conv = torch.randn((slots, query + 4, 2560), dtype=torch.bfloat16, device=device)[:, : query + 2]
    ssm = torch.randn((slots, 13, 128, 128), dtype=torch.float32, device=device)[:, :12]
    ids = torch.full((1, query), -1, dtype=torch.int32, device=device)
    ids[0, :3] = torch.tensor([2, 0, 5], dtype=torch.int32, device=device)
    buffers = prepare(conv, ssm, ids, ssm_layout=layout)
    gather(conv, ssm, ids, buffers)
    expected = ssm[2].transpose(-1, -2) if layout == "VK" else ssm[2]
    torch.testing.assert_close(buffers["ssm"][0], expected, rtol=0, atol=0)
    torch.testing.assert_close(buffers["conv"][0], conv[2], rtol=0, atol=0)
    assert torch.count_nonzero(buffers["ssm"][1]).item() == 0
    assert torch.count_nonzero(buffers["ssm"][3:]).item() == 0

    original_conv, original_ssm = conv.clone(), ssm.clone()

    def update():
        gather(conv, ssm, ids, buffers)
        buffers["conv"].zero_()
        buffers["ssm"].zero_()
        scatter(conv, ssm, ids, buffers)

    if graph_mode:
        update()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            update()
        conv.copy_(original_conv)
        ssm.copy_(original_ssm)
        graph.replay()
    else:
        update()
    torch.npu.synchronize()
    expected_conv, expected_ssm = original_conv.clone(), original_ssm.clone()
    expected_conv[[2, 5]] = 0
    expected_ssm[[2, 5]] = 0
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(ssm, expected_ssm, rtol=0, atol=0)
