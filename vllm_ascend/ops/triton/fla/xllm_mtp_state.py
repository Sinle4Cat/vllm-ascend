# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _copy_mtp_states_kernel(
    NativeConv,
    NativeSsm,
    Conv,
    Ssm,
    Output,
    Indices,
    QueryStarts,
    Accepted,
    N: tl.constexpr,
    S: tl.constexpr,
    CONV_SIZE: tl.constexpr,
    SSM_SIZE: tl.constexpr,
    CONV_STRIDE: tl.constexpr,
    SSM_STRIDE: tl.constexpr,
    OUT_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    checkpoint = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base_slot = tl.load(Indices)
    slot = tl.load(Indices + checkpoint)
    length = tl.load(QueryStarts + 1) - tl.load(QueryStarts)
    accepted = tl.load(Accepted)
    live = (length == S) & (accepted >= 1) & (accepted <= S) & (base_slot > 0) & (base_slot < N)
    ssm_mask = (cols < SSM_SIZE) & live & (slot > 0) & (slot < N)
    ssm_value = tl.load(Ssm + checkpoint * SSM_SIZE + cols, mask=ssm_mask, other=0)
    tl.store(NativeSsm + slot * SSM_STRIDE + cols, ssm_value, mask=ssm_mask)
    if checkpoint == 0:
        conv_mask = (cols < CONV_SIZE) & live
        conv_value = tl.load(Conv + cols, mask=conv_mask, other=0)
        tl.store(NativeConv + base_slot * CONV_STRIDE + cols, conv_value, mask=conv_mask)
    # A captured verification branch must be a no-op when its stable metadata
    # is reset for padding/no-spec replay. NULL_BLOCK_ID zero stays untouched.
    tl.store(Output + checkpoint * OUT_SIZE + cols, 0, mask=(cols < OUT_SIZE) & ~live)


def copy_mtp_states_to_native(
    native_conv: torch.Tensor,
    native_ssm: torch.Tensor,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    output: torch.Tensor,
    indices: torch.Tensor,
    query_starts: torch.Tensor,
    accepted: torch.Tensor,
) -> None:
    ssm_size = ssm.numel() // ssm.shape[0]
    conv_size = conv.numel()
    out_size = output.numel() // ssm.shape[0]
    block = 4096
    _copy_mtp_states_kernel[(ssm.shape[0], triton.cdiv(max(ssm_size, conv_size, out_size), block))](
        native_conv,
        native_ssm,
        conv,
        ssm,
        output,
        indices,
        query_starts,
        accepted,
        N=native_ssm.shape[0],
        S=ssm.shape[0],
        CONV_SIZE=conv_size,
        SSM_SIZE=ssm_size,
        CONV_STRIDE=native_conv.stride(0),
        SSM_STRIDE=native_ssm.stride(0),
        OUT_SIZE=out_size,
        BLOCK=block,
    )
