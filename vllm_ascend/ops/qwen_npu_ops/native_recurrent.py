# SPDX-License-Identifier: Apache-2.0
"""Native AscendC recurrence with compact queries and full VK history tables."""

import ctypes
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch


class NativeVkRecurrent:
    def __init__(self, payload):
        folder = Path(payload).resolve()
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("abi") != "qwen-native-vk-recurrent-v1" or manifest.get("status") != "COMPILED":
            raise ValueError("A compiled native VK recurrence payload is required")
        # Legacy manifests only declared the original Q1/Q8/Q9 host contract.
        supported = manifest.get("supported_queries", [1, 8, 9])
        if (
            not isinstance(supported, list)
            or not supported
            or any(type(width) is not int or width not in (1, 4, 8, 9) for width in supported)
            or len(set(supported)) != len(supported)
        ):
            raise ValueError("Invalid native VK recurrence query capabilities")
        self.supported_queries = tuple(supported)
        target = manifest["target"]
        if target != dict(machine=platform.machine(), torch=version("torch"), torch_npu=version("torch-npu")):
            raise ValueError("Native VK recurrence runtime identity mismatch")
        library = (folder / manifest["library"]).resolve()
        if library.parent != folder or hashlib.sha256(library.read_bytes()).hexdigest() != manifest["library_sha256"]:
            raise ValueError("Native VK recurrence library identity mismatch")
        self.library = ctypes.CDLL(str(library))
        self.library.call.restype = None
        self.library.call.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_uint32] * 6 + [ctypes.c_float, ctypes.c_void_p]
        self.identity = dict(manifest=manifest, path=str(folder))

    def __call__(
        self,
        query,
        key,
        value,
        state,
        *,
        beta,
        scale=1.0,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens=None,
        g=None,
        gk=None,
    ):
        tokens = query.shape[0]
        batches = actual_seq_lengths.numel() - 1
        if batches < 1 or ssm_state_indices.numel() % batches:
            raise ValueError("Full history indices must contain B x Q entries")
        width = ssm_state_indices.numel() // batches
        if width not in self.supported_queries:
            raise ValueError("Native VK recurrence payload does not declare this history width")
        if query.shape != (tokens, 4, 128) or key.shape != query.shape or value.shape != (tokens, 12, 128):
            raise ValueError("Native VK recurrence requires the TP4 Qwen head geometry")
        if state.shape[1:] != (12, 128, 128) or state.stride()[1:] != (16384, 128, 1):
            raise ValueError("Native VK recurrence requires contiguous inner VK state pages")
        if state.device.type != "npu" or state.dtype != torch.float32 or g is None or gk is not None:
            raise ValueError("Native VK recurrence requires FP32 VK state and scalar gates")
        tensors = (query, key, value, beta, actual_seq_lengths, ssm_state_indices, g)
        if any(t.device != state.device or not t.is_contiguous() for t in tensors):
            raise ValueError("Native VK recurrence inputs must be contiguous on the state device")
        if any(t.dtype != torch.bfloat16 for t in (query, key, value, beta)) or g.dtype != torch.float32:
            raise ValueError("Native VK recurrence requires BF16 Q/K/V/beta and FP32 gates")
        if actual_seq_lengths.dtype != torch.int32 or ssm_state_indices.dtype != torch.int32:
            raise ValueError("Native VK recurrence metadata must be INT32")
        if beta.shape != (tokens, 12) or g.shape != beta.shape:
            raise ValueError("Native VK recurrence gate shapes must match token and value heads")
        if num_accepted_tokens is not None and (
            num_accepted_tokens.shape != (batches,)
            or num_accepted_tokens.dtype != torch.int32
            or num_accepted_tokens.device != state.device
            or not num_accepted_tokens.is_contiguous()
        ):
            raise ValueError("Native VK recurrence accepted counts must be contiguous INT32[B]")
        output = torch.empty_like(value)
        if tokens:
            pointers = [
                t.data_ptr() if t is not None else 0
                for t in (
                    query,
                    key,
                    value,
                    beta,
                    state,
                    actual_seq_lengths,
                    ssm_state_indices,
                    g,
                    num_accepted_tokens,
                    output,
                )
            ]
            self.library.call(
                *pointers,
                tokens,
                batches,
                width,
                *state.stride()[:3],
                float(scale),
                torch.npu.current_stream().npu_stream,
            )
        torch.autograd.graph.increment_version(state)
        return output
