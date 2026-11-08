"""Configure the private R30 OPP before importing torch_npu or vLLM."""

import hashlib
import json
import os
from pathlib import Path

from vllm_ascend import envs


def read_identity(path):
    identity = json.loads(Path(path).read_text())
    vendor = Path(identity["vendor"]).resolve()
    for relative, expected in identity["payload"].items():
        source = (vendor / relative).resolve()
        if vendor not in source.parents or hashlib.sha256(source.read_bytes()).hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Prefill payload identity mismatch: {source}")
    return identity, vendor


def configure_prefill_environment(identity_path):
    """Call in the entry point before CANN initializes its operator registry."""
    identity, vendor = read_identity(identity_path)
    paths = [str(vendor), str(vendor.parent.parent), *envs.ASCEND_CUSTOM_OPP_PATH.split(os.pathsep)]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys(filter(None, paths)))
    return identity
