"""Explicit R30 prefill adapter with named KV or VK state banks.

The private OPP and matching torch extension are explicit inputs. This does
not modify global OPP registration or replace unsupported framework paths.
"""

import hashlib
import importlib.util
import os
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend import envs

from .prefill_env import read_identity


@triton.jit
def _state_io(bank, slot, dense, STRIDE: tl.constexpr, WIDTH: tl.constexpr, SCATTER: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(slot).to(tl.int64)
    valid = (page > 0) & (offsets < WIDTH)
    if SCATTER:
        values = tl.load(dense + offsets, mask=valid, other=0)
        tl.store(bank + page * STRIDE + offsets, values, mask=valid)
    else:
        values = tl.load(bank + page * STRIDE + offsets, mask=valid, other=0)
        tl.store(dense + offsets, values, mask=offsets < WIDTH)


def _copy_state(bank, slot, dense, width, *, scatter):
    if not bank[0].is_contiguous() or not dense.is_contiguous():
        raise ValueError("Prefill state needs contiguous pages; bank stride may be padded")
    _state_io[(triton.cdiv(width, 1024),)](bank, slot, dense, bank.stride(0), width, scatter, 1024)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GdnPrefill:
    """R30 single-sequence calls, serialized on the current model stream.

    KV input/output is copied without orientation changes. VK input/output
    retains the operator's internal conversion. Only the first three Conv history rows are modified by
    prefill; K8's speculative history rows must remain untouched.
    """

    def __init__(self, identity, extension, *, ssm_layout="VK"):
        if ssm_layout not in ("KV", "VK"):
            raise ValueError("Specify the Prefill SSM layout as KV or VK")
        self.ssm_layout = ssm_layout
        self.identity_path = Path(identity).resolve()
        self.extension_path = Path(extension).resolve()
        self.identity, self.vendor = read_identity(self.identity_path)
        if str(self.vendor) not in envs.ASCEND_CUSTOM_OPP_PATH.split(os.pathsep):
            raise RuntimeError(
                "Configure the R30 OPP before importing torch_npu or vLLM; "
                "late registration cannot initialize CANN tiling discovery"
            )
        spec = importlib.util.spec_from_file_location(self.extension_path.stem, self.extension_path)
        self.extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.extension)
        from fla_npu.ops.ascendc import _runtime

        opapi = self.vendor / "op_api/lib/libcust_opapi.so"
        libraries = [str(opapi), *[lib._name for lib in _runtime.runtime()._libraries[1:]]]
        self.extension.initialize(libraries, str(opapi))
        self._buffers = {}

    def prepare(self, device):
        key = str(device)
        if key not in self._buffers:
            self._buffers[key] = dict(
                conv=torch.zeros((1, 6, 2560), dtype=torch.bfloat16, device=device),
                ssm=torch.empty((1, 12, 128, 128), dtype=torch.float32, device=device),
                zero=torch.zeros(1, dtype=torch.int64, device=device),
            )
        return self._buffers[key]

    @staticmethod
    def supports(layer, meta, qkv, z, b, a):
        if (
            meta.num_prefills < 1
            or meta.num_decodes != 0
            or meta.num_spec_decodes != 0
            or meta.spec_sequence_masks is not None
        ):
            return False
        count = int(meta.num_actual_tokens)
        if count <= 0 or tuple(qkv.shape[1:]) != (2560,):
            return False
        if tuple(z.shape[1:]) != (12, 128) or tuple(b.shape[1:]) != (12,) or a.shape != b.shape:
            return False
        if any(x.dtype != torch.bfloat16 for x in (qkv, z, b, a)):
            return False
        conv, ssm = layer.kv_cache[:2]
        steps = getattr(layer, "num_spec", None)
        if (
            steps not in (7, 8)
            or tuple(conv.shape[1:]) != (3 + steps, 2560)
            or tuple(ssm.shape[1:]) != (12, 128, 128)
            or conv.dtype != torch.bfloat16
            or ssm.dtype != torch.float32
        ):
            return False
        prefill = meta.non_spec_prefill_metadata
        chunk = None if prefill is None else prefill.chunk
        cu = None if chunk is None else getattr(chunk, "cu_seqlens_host", None)
        if cu is None or len(cu) != meta.num_prefills + 1:
            return False
        ends = [int(x) for x in cu]
        if ends[0] != 0 or ends[-1] != count or any(y <= x for x, y in zip(ends, ends[1:])):
            return False
        indices, initial = meta.prefill_state_indices, meta.prefill_has_initial_state
        if (
            indices is None
            or initial is None
            or indices.numel() != meta.num_prefills
            or initial.numel() != meta.num_prefills
        ):
            return False
        return (
            layer.conv1d.bias is None
            and layer.norm.eps == 1e-6
            and layer.norm.norm_before_gate
            and layer.A_log.dtype == torch.float32
        )

    def run_sequence(self, qkv, z, b, a, weight, alog, bias, norm, conv, ssm, slot, has_initial, output):
        buffers = self.prepare(qkv.device)
        slot = slot.to(torch.int64).contiguous()
        initial = has_initial.to(torch.bool).contiguous()
        # Page copies preserve the explicitly selected orientation and stride.
        _copy_state(conv, slot, buffers["conv"], 3 * 2560, scatter=False)
        _copy_state(ssm, slot, buffers["ssm"], 12 * 128 * 128, scatter=False)
        values = [
            qkv.contiguous(),
            a.contiguous(),
            b.contiguous(),
            z.contiguous(),
            weight.contiguous(),
            alog.float().contiguous(),
            bias.float().contiguous(),
            buffers["conv"],
            buffers["ssm"],
            norm.contiguous(),
            buffers["zero"],
            buffers["zero"],
            buffers["zero"],
            buffers["zero"],
            initial,
            initial,
        ]
        if self.ssm_layout == "KV":
            self.extension.run(values, output, False, True)
        else:
            self.extension.run(values, output, False)
        _copy_state(conv, slot, buffers["conv"], 3 * 2560, scatter=True)
        _copy_state(ssm, slot, buffers["ssm"], 12 * 128 * 128, scatter=True)
        return output

    def __call__(self, layer, meta, qkv, z, b, a, output, weight):
        if not self.supports(layer, meta, qkv, z, b, a):
            raise ValueError("Unsupported R30 model prefill metadata")
        ends = [int(x) for x in meta.non_spec_prefill_metadata.chunk.cu_seqlens_host]
        for seq, (start, end) in enumerate(zip(ends, ends[1:])):
            self.run_sequence(
                qkv[start:end],
                z[start:end],
                b[start:end],
                a[start:end],
                weight,
                layer.A_log,
                layer.dt_bias,
                layer.norm.weight,
                *layer.kv_cache[:2],
                meta.prefill_state_indices[seq : seq + 1],
                meta.prefill_has_initial_state[seq : seq + 1],
                output[start:end],
            )
        return len(ends) - 1

    def status(self):
        providers = dict(self.extension.status()["providers"])
        opapi = str(self.vendor / "op_api/lib/libcust_opapi.so")
        if any(
            providers[name] != opapi
            for name in ("aclnnMegaGdnPrefillPr990", "aclnnMegaGdnPrefillPr990GetWorkspaceSize")
        ):
            raise RuntimeError(f"Wrong R30 provider: {providers}")
        return dict(
            ssm_layout=self.ssm_layout,
            identity_sha256=_sha256(self.identity_path),
            bridge_sha256=_sha256(self.extension_path),
            payload_sha256=self.identity["payload_identity_sha256"],
            runtime=dict(self.extension.status()),
        )
