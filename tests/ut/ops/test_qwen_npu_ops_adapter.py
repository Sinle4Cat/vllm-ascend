# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in Qwen adapters; no NPU arithmetic claims."""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / "vllm_ascend/ops/qwen_npu_ops"
WORKER = ROOT / "vllm_ascend/worker/qwen_npu_ops"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.device = "npu:0"

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def contiguous(self):
        return self

    def to(self, dtype):
        return self

    def float(self):
        return self

    def __getitem__(self, selection):
        if isinstance(selection, slice):
            return Tensor((len(range(*selection.indices(self.shape[0]))), *self.shape[1:]), self.dtype)
        raise AssertionError(selection)


class QwenAdapterContracts(unittest.TestCase):
    def setUp(self):
        # Isolate adapters from optional torch/vLLM dependencies and restore
        # every imported test module after each case, including under pytest.
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        root = types.ModuleType("vllm_ascend")
        root.__path__ = [str(ROOT / "vllm_ascend")]
        ops = types.ModuleType("vllm_ascend.ops")
        ops.__path__ = [str(ROOT / "vllm_ascend/ops")]
        ops.gdn = types.SimpleNamespace(_get_packed_conv_weights=lambda _: Tensor((4, 2560)))
        worker = types.ModuleType("vllm_ascend.worker")
        worker.__path__ = [str(ROOT / "vllm_ascend/worker")]
        torch = types.ModuleType("torch")
        torch.bfloat16, torch.float32, torch.int32 = "bf16", "fp32", "i32"
        torch.ones = lambda count, **kwargs: Tensor((count,), "i32")
        torch.empty = lambda shape, **kwargs: Tensor(shape, kwargs["dtype"])
        torch.npu = types.SimpleNamespace(is_current_stream_capturing=lambda: True)
        utils = types.SimpleNamespace(
            triton=types.SimpleNamespace(jit=lambda f: f), tl=types.SimpleNamespace(constexpr=object())
        )
        sys.modules.update(
            {
                "vllm_ascend": root,
                "vllm_ascend.ops": ops,
                "vllm_ascend.ops.gdn": ops.gdn,
                "vllm_ascend.worker": worker,
                "torch": torch,
                "qwen_npu_ops": types.SimpleNamespace(GdnDecode=Mock()),
                "vllm.triton_utils": utils,
            }
        )
        self.envs = load("vllm_ascend.envs", ROOT / "vllm_ascend/envs.py")
        root.envs = self.envs
        package = types.ModuleType("vllm_ascend.ops.qwen_npu_ops")
        package.__path__ = [str(ADAPTER)]
        sys.modules[package.__name__] = package
        self.prefill_env = load(package.__name__ + ".prefill_env", ADAPTER / "prefill_env.py")
        self.prefill = load(package.__name__ + ".prefill", ADAPTER / "prefill.py")

    def identity(self, folder, relative="op_api/lib/libcust_opapi.so"):
        vendor = folder / "opp/vendors/qwen"
        source = vendor / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"identity fixture")
        identity = folder / "identity.json"
        identity.write_text(
            json.dumps(
                {
                    "vendor": str(vendor),
                    "payload": {relative: {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}},
                }
            )
        )
        return identity, vendor, source

    def test_disabled_by_default_and_strict_enable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.envs.VLLM_ASCEND_QWEN_PREFILL_ENABLE)
            self.assertIsNone(self.envs.VLLM_ASCEND_QWEN_DECODE_PAYLOAD)
        for value, expected in (("0", False), ("1", True)):
            with patch.dict(os.environ, {"VLLM_ASCEND_QWEN_PREFILL_ENABLE": value}):
                self.assertEqual(self.envs.VLLM_ASCEND_QWEN_PREFILL_ENABLE, expected)
        with patch.dict(os.environ, {"VLLM_ASCEND_QWEN_PREFILL_ENABLE": "yes"}), self.assertRaises(ValueError):
            _ = self.envs.VLLM_ASCEND_QWEN_PREFILL_ENABLE

    def test_private_provider_precedes_existing_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            identity, vendor, _ = self.identity(Path(folder))
            with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "existing"}):
                self.prefill_env.configure_prefill_environment(identity)
                self.prefill_env.configure_prefill_environment(identity)
                self.assertEqual(
                    os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep),
                    [str(vendor.resolve()), str(vendor.parent.parent.resolve()), "existing"],
                )

    def test_payload_tamper_rejected_before_environment_change(self):
        with tempfile.TemporaryDirectory() as folder:
            identity, _, source = self.identity(Path(folder))
            source.write_bytes(b"changed")
            with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "existing"}):
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    self.prefill_env.configure_prefill_environment(identity)
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], "existing")

    def test_payload_outside_vendor_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            identity, _, _ = self.identity(Path(folder), "../../../foreign.so")
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                self.prefill_env.configure_prefill_environment(identity)

    def test_prefill_contracts_keep_k3_k7_k8_and_real_boundaries(self):
        layer = types.SimpleNamespace(
            num_spec=8,
            kv_cache=[Tensor((15, 11, 2560)), Tensor((15, 12, 128, 128), "fp32")],
            conv1d=types.SimpleNamespace(bias=None),
            A_log=Tensor((12,), "fp32"),
            norm=types.SimpleNamespace(eps=1e-6, norm_before_gate=True),
        )
        meta = types.SimpleNamespace(
            num_prefills=2,
            num_decodes=0,
            num_spec_decodes=0,
            spec_sequence_masks=None,
            num_actual_tokens=129,
            non_spec_prefill_metadata=types.SimpleNamespace(chunk=types.SimpleNamespace(cu_seqlens_host=(0, 1, 129))),
            prefill_state_indices=Tensor((2,), "i64"),
            prefill_has_initial_state=Tensor((2,), "bool"),
        )
        values = [Tensor((129, 2560)), Tensor((129, 12, 128)), Tensor((129, 12)), Tensor((129, 12))]
        supports = self.prefill.GdnPrefill.supports
        for steps in (3, 7, 8):
            with (
                patch.object(layer, "num_spec", steps),
                patch.object(layer.kv_cache[0], "shape", (15, steps + 3, 2560)),
            ):
                self.assertTrue(supports(layer, meta, *values))
        for key, value in [("num_decodes", 1), ("num_spec_decodes", 1), ("spec_sequence_masks", object())]:
            with patch.object(meta, key, value):
                self.assertFalse(supports(layer, meta, *values))
        for ends in ((0, 1, 128), (0, 0, 129), (1, 2, 129)):
            with patch.object(meta.non_spec_prefill_metadata.chunk, "cu_seqlens_host", ends):
                self.assertFalse(supports(layer, meta, *values))

    def test_compact_decode_keeps_tokens_and_complete_history(self):
        prefix = "vllm_ascend.ops.qwen_npu_ops"
        load(prefix + ".state_bridge", ADAPTER / "state_bridge.py")
        chain_module = load(prefix + ".kv_chain", ADAPTER / "kv_chain.py")
        for query in (4, 8, 9):
            for rows in (0, 3, 4 * query):
                decode = Mock(return_value=Tensor((rows, 12, 128)))
                chain = chain_module.KvGdnChain(decode, Mock())
                buffers = dict(
                    conv=Tensor((4 * query, query + 2, 2560)),
                    ssm=Tensor((4 * query, 12, 128, 128)),
                    local=Tensor((4, query)),
                    capacity=Tensor((1,)),
                )
                chain.bridges[(4, query)] = buffers
                chain.workspaces[(4, query)] = object()
                layer = types.SimpleNamespace(
                    num_spec=query - 1,
                    kv_cache=[Tensor((50, query + 2, 2560)), Tensor((50, 12, 128, 128))],
                    A_log=Tensor((12,)),
                    dt_bias=Tensor((12,)),
                    norm=types.SimpleNamespace(weight=Tensor((128,))),
                )
                starts, indices, accepted = Tensor((5,)), Tensor((4, query)), Tensor((4,))
                with patch.object(chain_module, "gather") as gather, patch.object(chain_module, "scatter") as scatter:
                    result = chain.decode_group(
                        layer,
                        Tensor((rows, 2560)),
                        Tensor((rows, 12, 128)),
                        Tensor((rows, 12)),
                        Tensor((rows, 12)),
                        starts,
                        indices,
                        accepted,
                    )
                    self.assertEqual(result.shape[0], rows)
                    self.assertEqual(decode.call_args.args[0].shape[0], rows)
                    self.assertIs(decode.call_args.args[10], starts)
                    self.assertIs(decode.call_args.args[11], accepted)
                    self.assertIs(gather.call_args.args[2], indices)
                    self.assertIs(scatter.call_args.args[2], indices)
                    self.assertEqual(chain.calls[f"r22_capture_B4_Q{query}"], 1)

    def test_decode_workspace_is_bounded_by_configured_batch(self):
        prefix = "vllm_ascend.ops.qwen_npu_ops"
        load(prefix + ".state_bridge", ADAPTER / "state_bridge.py")
        chain_module = load(prefix + ".kv_chain", ADAPTER / "kv_chain.py")
        decode = Mock(buckets=tuple((batch, query) for batch in range(1, 33) for query in (4, 8, 9)))
        chain = chain_module.KvGdnChain(decode, Mock())
        conv, state = Tensor((1, 11, 2560)), Tensor((1, 12, 128, 128), "fp32")
        with patch.object(chain_module, "prepare"):
            chain.prepare(conv, state, max_batch=8)
            chain.prepare(conv, state, max_batch=8)
            self.assertEqual(set(chain.bridges), {(batch, 9) for batch in range(1, 9)})
            self.assertEqual(decode.allocate_workspace.call_count, 8)
            with self.assertRaisesRegex(ValueError, "Missing R22"):
                chain.prepare(conv, state, max_batch=33)

    def test_kv_q4_prepares_six_row_history_before_capture(self):
        prefix = "vllm_ascend.ops.qwen_npu_ops"
        load(prefix + ".state_bridge", ADAPTER / "state_bridge.py")
        module = load(prefix + ".kv_chain", ADAPTER / "kv_chain.py")
        decode = Mock(buckets=tuple((batch, 4) for batch in range(1, 33)))
        chain = module.KvGdnChain(decode, Mock())
        with patch.object(module, "prepare") as prepare:
            chain.prepare(Tensor((1, 6, 2560)), Tensor((1, 12, 128, 128), "fp32"), max_batch=32)
        self.assertEqual(set(chain.bridges), {(batch, 4) for batch in range(1, 33)})
        self.assertEqual(decode.allocate_workspace.call_count, 32)
        self.assertEqual(prepare.call_args.args[2].shape, (32, 4))

    def test_bank_workspace_cache_does_not_skip_real_bank_binding(self):
        prefix = "vllm_ascend.ops.qwen_npu_ops"
        load(prefix + ".state_bridge", ADAPTER / "state_bridge.py")
        module = load(prefix + ".bank_chain", ADAPTER / "bank_chain.py")
        decode = Mock(buckets=tuple((batch, 8) for batch in range(1, 9)))
        chain = module.BankKvGdnChain(decode, Mock())
        dummy = Tensor((1, 10, 2560)), Tensor((1, 12, 128, 128), "fp32")
        real = Tensor((724, 10, 2560)), Tensor((724, 12, 128, 128), "fp32")
        chain.prepare(*dummy, max_batch=8)
        chain.prepare(*real, max_batch=8)
        self.assertEqual(decode.prepare_bank.call_count, 2)
        self.assertIs(decode.prepare_bank.call_args.args[0], real[0])
        self.assertEqual(decode.allocate_workspace.call_count, 8)
        self.assertFalse(chain.bridges)

    def test_bank_decode_passes_global_ids_true_boundaries_and_accepted(self):
        prefix = "vllm_ascend.ops.qwen_npu_ops"
        bridge = load(prefix + ".state_bridge", ADAPTER / "state_bridge.py")
        module = load(prefix + ".bank_chain", ADAPTER / "bank_chain.py")
        for query in (8, 9):
            decode = Mock(return_value=Tensor((3, 12, 128)))
            chain = module.BankKvGdnChain(decode, Mock())
            chain.workspaces[4, query] = object()
            conv, ssm = Tensor((724, query+2, 2560)), Tensor((724, 12, 128, 128), "fp32")
            layer = types.SimpleNamespace(num_spec=query-1, kv_cache=[conv, ssm], A_log=Tensor((12,)),
                                          dt_bias=Tensor((12,)), norm=types.SimpleNamespace(weight=Tensor((128,))))
            starts, indices, accepted = Tensor((5,)), Tensor((4, query)), Tensor((4,))
            with patch.object(bridge, "gather") as gather, patch.object(bridge, "scatter") as scatter:
                chain.decode_group(layer, Tensor((3, 2560)), Tensor((3, 12, 128)),
                                   Tensor((3, 12)), Tensor((3, 12)), starts, indices, accepted)
                gather.assert_not_called()
                scatter.assert_not_called()
            args = decode.call_args.args
            self.assertIs(args[5], conv)
            self.assertIs(args[8], ssm)
            self.assertIs(args[9], indices)
            self.assertIs(args[10], starts)
            self.assertIs(args[11], accepted)
            self.assertIs(args[13], decode.metadata_for_bank.return_value)
            self.assertEqual(chain.calls["r22_bank_groups"], 1)

    def test_native_vk_manifest_declares_q4_without_relabeling_legacy(self):
        module = load("qwen_native_vk_contract", ADAPTER / "native_recurrent.py")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "native.so"
            library.write_bytes(b"CPU fixture; never loaded")
            manifest = dict(
                abi="qwen-native-vk-recurrent-v1",
                status="COMPILED",
                target=dict(machine="aarch64", torch="fixture", torch_npu="fixture"),
                library=library.name,
                library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            )

            def construct(capability=None):
                data = dict(manifest)
                if capability is not None:
                    data["supported_queries"] = capability
                (root / "manifest.json").write_text(json.dumps(data))
                return module.NativeVkRecurrent(root)

            with (
                patch.object(module, "version", return_value="fixture"),
                patch.object(module.platform, "machine", return_value="aarch64"),
                patch.object(module.ctypes, "CDLL") as loader,
            ):
                legacy = construct()
                self.assertEqual(legacy.supported_queries, (1, 8, 9))
                with self.assertRaisesRegex(ValueError, "does not declare"):
                    legacy(
                        Tensor((1, 4, 128)), Tensor((1, 4, 128)), Tensor((1, 12, 128)),
                        Tensor((5, 12, 128, 128), "fp32"), beta=Tensor((1, 12)),
                        actual_seq_lengths=Tensor((2,), "i32"), ssm_state_indices=Tensor((4,), "i32"),
                    )
                loader.return_value.call.assert_not_called()
                self.assertEqual(construct([1, 4, 8, 9]).supported_queries, (1, 4, 8, 9))
                for bad in ([], [1, 4, 4], [1, True], [1, 3], "1,4"):
                    before = loader.call_count
                    with self.assertRaisesRegex(ValueError, "capabilities"):
                        construct(bad)
                    self.assertEqual(loader.call_count, before)

    def test_rope_uses_r01_for_supported_inputs_and_native_for_others(self):
        rope = Mock()
        native = Mock()
        sys.modules["qwen_npu_ops"].Rope = Mock(return_value=rope)
        sys.modules["vllm.utils.torch_utils"] = types.SimpleNamespace(direct_register_custom_op=Mock())
        sys.modules["torch"].Tensor = Tensor
        sys.modules["vllm_ascend.patch.worker.patch_qwen3_5"] = types.SimpleNamespace(
            Qwen3NextAttention=Mock, _uses_multimodal_rope=Mock(return_value=True), native_qwen_mrope=native
        )
        module = load("qwen_rope_contract", ADAPTER / "rope.py")
        layer = Mock(
            num_heads=6,
            num_kv_heads=1,
            head_dim=256,
            attn_output_gate=True,
            config=types.SimpleNamespace(rms_norm_eps=1e-6),
            rotary_emb=types.SimpleNamespace(mrope_section=(11, 11, 10), mrope_interleaved=True, rotary_dim=64),
        )
        layers = [(f"model.layers.{i}.self_attn", layer) for i in range(16)]
        adapter = module.QwenRopeAdapter("fixture", Mock(named_modules=Mock(return_value=layers)))
        self.assertTrue(layer._qwen_rope_r01)
        args = Tensor((72, 3584)), *[Mock() for _ in range(4)]
        rope.supports.return_value = True
        self.assertIs(adapter.execute(*args), rope.return_value)
        self.assertEqual(adapter.snapshot()["capture_hits"], {"M72": 1})
        args = Tensor((32, 3584)), *args[1:]
        self.assertIs(adapter.execute(*args), rope.return_value)
        self.assertEqual(adapter.snapshot()["capture_hits"], {"M72": 1, "M32": 1})
        rope.supports.return_value = False
        args = Tensor((73, 3584)), *args[1:]
        self.assertIs(adapter.execute(*args), native.return_value)
        self.assertEqual(native.call_args.args, args)
        self.assertEqual(adapter.snapshot()["calls"], {"r01_M72": 1, "r01_M32": 1, "native_M73": 1})
        args = Tensor((128, 3584)), *args[1:]
        self.assertIs(adapter.execute(*args), native.return_value)
        self.assertEqual(native.call_args.args, args)
        self.assertEqual(adapter.snapshot()["calls"]["native_M128"], 1)

    def test_vk_mixed_contract_preserves_source_metadata(self):
        module = load("vllm_ascend.ops.qwen_npu_ops.vk_chain", ADAPTER / "vk_chain.py")
        prefill = Mock()
        prefill.supports.return_value = True
        chain = module.VkGdnChain(prefill)
        meta = types.SimpleNamespace(
            num_actual_tokens=137,
            num_prefills=2,
            num_decodes=1,
            num_decode_tokens=1,
            num_spec_decodes=2,
            spec_sequence_masks=object(),
            spec_token_indx=Tensor((7,)),
            non_spec_token_indx=Tensor((130,)),
            spec_decode_metadata=object(),
            non_spec_decode_metadata=object(),
        )
        values = [Tensor((144, 2560)), Tensor((144, 12, 128)), Tensor((144, 12)), Tensor((144, 12))]
        self.assertTrue(chain.supports(object(), meta, *values))
        narrowed = prefill.supports.call_args.args[1]
        self.assertEqual(narrowed.num_actual_tokens, 129)
        self.assertEqual(narrowed.num_decodes, 0)
        self.assertIsNone(narrowed.spec_sequence_masks)
        self.assertEqual(meta.num_actual_tokens, 137)
        self.assertEqual(meta.num_decodes, 1)
        self.assertIsNotNone(meta.spec_sequence_masks)
        with patch.object(meta, "non_spec_token_indx", Tensor((129,))):
            self.assertFalse(chain.supports(object(), meta, *values))
        with patch.object(meta, "num_decode_tokens", 2):
            self.assertFalse(chain.supports(object(), meta, *values))

    def test_kv_worker_rejects_disabled_prefill_before_loading(self):
        # Import only the subclass; the heavy model-loading parent stays mocked.
        parent = types.SimpleNamespace(QwenPrefillWorker=type("Parent", (), {}))
        sys.modules["vllm_ascend.worker.qwen_npu_ops.prefill_worker"] = parent
        prefix = "vllm_ascend.ops.qwen_npu_ops"
        load(prefix + ".state_bridge", ADAPTER / "state_bridge.py")
        load(prefix + ".kv_chain", ADAPTER / "kv_chain.py")
        module = load("qwen_kv_worker_contract", WORKER / "kv_worker.py")
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_QWEN_PREFILL_ENABLE": "0"}),
            self.assertRaisesRegex(ValueError, "requires enabled R30"),
        ):
            module.QwenKvWorker().load_model()


if __name__ == "__main__":
    unittest.main()
