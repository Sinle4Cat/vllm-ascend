"""Selected-row IO between vLLM state pages and the R22 K,V state ABI.

vLLM-Ascend 0.30 stores GDN state as V,K. R22 consumes K,V. Equal head
dimensions hide this difference from shape checks, so callers must name the
framework layout explicitly. Keep this adapter outside the standalone API.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _copy_rows(
    bank,
    ids,
    dense,
    stride: tl.constexpr,
    width: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(ids + row).to(tl.int64)
    valid = (slot > 0) & (offset < width)
    source_offset = offset
    if TRANSPOSE:
        source_offset = offset // 16384 * 16384 + offset % 128 * 128 + offset // 128 % 128
    address = bank + slot * stride + source_offset
    if SCATTER:
        value = tl.load(dense + row * width + offset, mask=valid, other=0)
        tl.store(address, value, mask=valid)
    else:
        value = tl.load(address, mask=valid, other=0)
        tl.store(dense + row * width + offset, value, mask=offset < width)


def _transfer(bank, indices, dense, *, transpose, scatter):
    assert bank[0].is_contiguous() and dense.is_contiguous()
    assert indices.is_contiguous() and indices.numel() == dense.shape[0]
    width = dense[0].numel()
    _copy_rows[(indices.numel(), triton.cdiv(width, 1024))](
        bank, indices, dense, bank.stride(0), width, transpose, scatter, 1024
    )


def prepare(bank_conv, bank_ssm, indices, *, ssm_layout):
    """Allocate before capture; retain one bridge per in-flight batch/Graph.

    Slot zero is vLLM's null block. Dense local slot zero is a valid R22 slot.
    The adapter maps null/padding slots to -1 and never writes their pages.
    """
    if ssm_layout not in ("VK", "KV"):
        raise ValueError("Specify the framework SSM layout as VK or KV")
    if tuple(bank_ssm.shape[1:]) != (12, 128, 128) or bank_ssm.dtype != torch.float32:
        raise ValueError("R22 bridge requires FP32 state [slots,12,128,128]")
    rows = indices.numel()
    conv = torch.empty((rows, *bank_conv.shape[1:]), dtype=bank_conv.dtype, device=bank_conv.device)
    ssm = torch.empty((rows, *bank_ssm.shape[1:]), dtype=bank_ssm.dtype, device=bank_ssm.device)
    local = torch.arange(rows, dtype=torch.int32, device=indices.device).view_as(indices)
    return dict(
        conv=conv,
        ssm=ssm,
        local_base=local,
        local=torch.empty_like(local),
        capacity=torch.tensor([rows], dtype=torch.int32, device=indices.device),
        transpose_ssm=ssm_layout == "VK",
    )


def gather(bank_conv, bank_ssm, indices, buffers):
    _transfer(bank_conv, indices, buffers["conv"], transpose=False, scatter=False)
    _transfer(bank_ssm, indices, buffers["ssm"], transpose=buffers["transpose_ssm"], scatter=False)
    buffers["local"].copy_(torch.where(indices > 0, buffers["local_base"], -1))


def scatter(bank_conv, bank_ssm, indices, buffers):
    _transfer(bank_conv, indices, buffers["conv"], transpose=False, scatter=True)
    _transfer(bank_ssm, indices, buffers["ssm"], transpose=buffers["transpose_ssm"], scatter=True)
