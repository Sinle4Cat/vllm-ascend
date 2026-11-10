# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Qwen W8A8 decode routing."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel


class TestQwenQuantMatmulAdapter(unittest.TestCase):
    def setUp(self):
        self.forward_context = Mock()
        self.native = Mock(return_value=sentinel.native)
        self.custom = Mock(return_value=sentinel.custom)
        op = types.SimpleNamespace(try_run=self.custom, manifest={"library_sha256": "fixture"})
        modules = {
            "torch": types.SimpleNamespace(
                bfloat16=sentinel.bfloat16,
                npu=types.SimpleNamespace(is_current_stream_capturing=lambda: True),
            ),
            "torch_npu": types.SimpleNamespace(npu_quant_matmul=self.native),
            "qwen_npu_ops.quant_matmul": types.SimpleNamespace(QuantMatmul=lambda _: op, WEIGHT_SHAPES=()),
            "vllm.forward_context": types.SimpleNamespace(get_forward_context=self.forward_context),
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/qwen_npu_ops/quant_matmul.py"
        spec = importlib.util.spec_from_file_location("qwen_quant_matmul_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.adapter = module.QuantMatmulAdapter("fixture")
        self.args = (
            types.SimpleNamespace(shape=(64, 5120)),
            types.SimpleNamespace(shape=(5120, 4096)),
            sentinel.scale,
            sentinel.token,
        )

    def test_decode_hits_custom_and_records_capture(self):
        self.forward_context.return_value.attn_metadata = {"gdn": types.SimpleNamespace(num_prefills=0)}
        self.assertIs(self.adapter.execute(*self.args), sentinel.custom)
        self.assertEqual(self.adapter.snapshot()["capture_hits"], {"64,5120,4096": 1})
        self.native.assert_not_called()

    def test_decode_unsupported_shape_uses_native(self):
        self.forward_context.return_value.attn_metadata = types.SimpleNamespace(num_prefill_tokens=0)
        self.custom.return_value = None
        self.assertIs(self.adapter.execute(*self.args), sentinel.native)
        self.native.assert_called_once_with(
            *self.args[:3], pertoken_scale=sentinel.token, bias=None, output_dtype=sentinel.bfloat16
        )

    def test_prefill_mixed_or_unknown_uses_native(self):
        for metadata in (
            {"gdn": types.SimpleNamespace(num_prefills=0), "attention": types.SimpleNamespace(num_prefill_tokens=1)},
            {"gdn": types.SimpleNamespace(num_prefills=2)},
            None,
        ):
            with self.subTest(metadata=metadata):
                self.forward_context.return_value.attn_metadata = metadata
                self.assertIs(self.adapter.execute(*self.args), sentinel.native)
        self.forward_context.side_effect = AssertionError
        self.assertIs(self.adapter.execute(*self.args), sentinel.native)
        self.custom.assert_not_called()
