# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch/state contracts only; arithmetic is covered by NPU acceptance."""

import json
import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import test_qwen_npu_ops_adapter as contracts

ADAPTER, Tensor, load = contracts.ADAPTER, contracts.Tensor, contracts.load


class Tiny(Tensor):
    def storage_offset(self):
        return 0

    def float(self):
        return Tiny(self.shape, "fp32")

    def transpose(self, x, y):
        shape = list(self.shape)
        shape[x], shape[y] = shape[y], shape[x]
        return Tiny(tuple(shape), self.dtype)

    def reshape(self, *shape):
        return Tiny(shape, self.dtype)

    def view_as(self, other):
        return Tiny(other.shape, self.dtype)

    def copy_(self, other):
        self.copied_from = other
        return self

    def __getitem__(self, selection):
        value = super().__getitem__(selection)
        return Tiny(value.shape, value.dtype)


class StagedPrefillContracts(unittest.TestCase):
    setUp = contracts.QwenAdapterContracts.setUp

    def make_adapter(self, layout="KV"):
        module = load("vllm_ascend.ops.qwen_npu_ops.staged_prefill", ADAPTER / "staged_prefill.py")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        vendor = Path(temp.name).resolve()
        identity = vendor / "identity.json"
        identity.write_text(
            json.dumps(
                dict(
                    schema="qwen-chunk-gdn-runtime-identity/1",
                    vendor=str(vendor),
                    entry_points=["full", "prepare", "h", "o"],
                )
            )
        )
        self.events = []
        self.ops = [Mock(name=name) for name in ("prepare", "h", "o")]

        def constructor(index):
            def select(*args, **kwargs):
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)[0], str(vendor))
                self.assertTrue(kwargs["allow_experimental"])
                return self.ops[index]

            return select

        api = sys.modules["qwen_npu_ops"]
        api.ChunkGdnPrepare, api.ChunkGdnH, api.ChunkGdnO = [constructor(i) for i in range(3)]
        previous = os.pathsep.join((str(vendor / "adn"), str(vendor)))
        with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": previous}):
            adapter = module.ThreeStagePrefill(identity, ssm_layout=layout)
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], previous)
        return module, adapter, identity

    def configure_ops(self):
        def prepare(*args, **kwargs):
            self.events.append("prepare")
            self.assertEqual(args[3].dtype, "fp32")
            self.assertEqual(args[4].dtype, "fp32")
            self.assertEqual(args[3].shape, (1, 12, 7))
            self.assertIs(kwargs["cu_seqlens"], self.cu)
            return (
                Tiny((1, 12, 7), "fp32"),
                Tiny((1, 12, 7, 128)),
                Tiny((1, 12, 7, 128)),
                Tiny((1, 12, 7, 64)),
                args[0],
                args[1],
                None,
                None,
                None,
            )

        def h(*args, **kwargs):
            self.events.append("h")
            self.assertEqual(kwargs["initial_state"].dtype, "fp32")
            self.assertEqual(kwargs["state_layout"], "KV")
            return Tiny((1, 2, 12, 128, 128)), Tiny((1, 12, 7, 128)), Tiny((2, 12, 128, 128), "fp32")

        def o(*args, **kwargs):
            self.events.append("o")
            self.assertEqual(kwargs["scale"], 128**-0.5)
            return Tiny((1, 12, 7, 128))

        for op, body in zip(self.ops, (prepare, h, o)):
            op.side_effect = body

    def test_only_three_explicit_stages_and_fp32_state(self):
        _, adapter, _ = self.make_adapter()
        self.cu = [0, 2, 7]
        self.configure_ops()
        output, state = adapter.forward(
            Tiny((1, 4, 7, 128)),
            Tiny((1, 4, 7, 128)),
            Tiny((1, 12, 7, 128)),
            Tiny((1, 7, 12), "fp32"),
            Tiny((1, 7, 12)),
            Tiny((2, 12, 128, 128), "fp32"),
            self.cu,
        )
        self.assertEqual(self.events, ["prepare", "h", "o"])
        self.assertEqual(output.shape, (1, 7, 12, 128))
        self.assertEqual(state.dtype, "fp32")
        self.assertEqual(adapter.calls, Counter(prepare=1, h=1, o=1))

    def test_bf16_state_is_rejected_before_any_stage(self):
        _, adapter, _ = self.make_adapter()
        with self.assertRaisesRegex(ValueError, "FP32 SSM"):
            adapter.forward(None, None, None, None, None, Tiny((1, 12, 128, 128)), [0, 7])
        for op in self.ops:
            op.assert_not_called()

    def test_missing_stage_vendor_is_rejected_without_environment_change(self):
        module, _, identity = self.make_adapter()
        with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "other"}):
            with self.assertRaisesRegex(ValueError, "before device initialization"):
                module.ThreeStagePrefill(identity)
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], "other")

    def test_ragged_boundaries_preserved_and_invalid_or_padded_groups_rejected(self):
        module, _, _ = self.make_adapter()
        cu = [0, 2, 7]
        meta = types.SimpleNamespace(
            num_prefills=2,
            non_spec_prefill_metadata=types.SimpleNamespace(chunk=types.SimpleNamespace(cu_seqlens_host=cu)),
        )
        self.assertIs(module.ThreeStagePrefill._boundaries(meta, 7), cu)
        for malformed in ([0, 2, 8], [0, 0, 7], [1, 2, 7], [0, 7]):
            meta.non_spec_prefill_metadata.chunk.cu_seqlens_host = malformed
            with self.assertRaises(ValueError):
                module.ThreeStagePrefill._boundaries(meta, 7)

    def test_mixed_decode_prefix_is_excluded_from_batched_stage_group(self):
        prefix = "vllm_ascend.ops.qwen_npu_ops"
        load(prefix + ".state_bridge", ADAPTER / "state_bridge.py")
        module = load(prefix + ".kv_chain", ADAPTER / "kv_chain.py")
        sys.modules["torch"].zeros_like = lambda value: Tiny(value.shape, value.dtype)
        prefill = Mock()
        chain = module.KvGdnChain(Mock(), prefill)
        chain.decode_group = Mock(return_value=Tiny((2, 12, 128)))
        cm = types.SimpleNamespace(query_start_loc=Tensor((3,)))
        meta = types.SimpleNamespace(
            num_actual_tokens=9,
            spec_sequence_masks=None,
            num_prefills=2,
            num_decodes=2,
            num_decode_tokens=2,
            non_spec_decode_metadata=types.SimpleNamespace(causal_conv1d=cm),
            non_spec_state_indices_tensor=Tiny((2,)),
        )
        layer = types.SimpleNamespace(_qwen_ssm_layout="KV")
        chain(layer, meta, Tiny((9, 2560)), Tiny((9, 12, 128)), Tiny((9, 12)), Tiny((9, 12)), Tiny((9, 12, 128)))
        prefill.run_group.assert_called_once()
        self.assertIs(prefill.run_group.call_args.args[1], meta)
        self.assertEqual(prefill.run_group.call_args.args[2].shape, (7, 2560))
        self.assertEqual(chain.decode_group.call_args.args[1].shape, (2, 2560))
        self.assertEqual(chain.calls["three_stage_prefill_sequences"], 2)


if __name__ == "__main__":
    unittest.main()
