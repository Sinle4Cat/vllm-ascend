"""R30-only validation worker; retain the installed decode and RoPE implementations.

Use the same wrapper with VLLM_ASCEND_QWEN_PREFILL_ENABLE=0 for the before arm. All route
counters execute in the custom-op body, outside Dynamo-traced forward methods.
"""

import json
import types
from collections import Counter
from pathlib import Path

import torch
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend import envs
from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops.qwen_npu_ops.native_recurrent import NativeVkRecurrent
from vllm_ascend.ops.qwen_npu_ops.prefill import GdnPrefill
from vllm_ascend.ops.qwen_npu_ops.vk_chain import VkGdnChain
from vllm_ascend.worker.worker import NPUWorker


class QwenPrefillWorker(NPUWorker):
    def load_model(self):
        if not all(
            (
                envs.VLLM_ASCEND_QWEN_PREFILL_IDENTITY,
                envs.VLLM_ASCEND_QWEN_PREFILL_BRIDGE,
                envs.VLLM_ASCEND_QWEN_OPS_RUN_DIR,
            )
        ):
            raise ValueError("Qwen workers require explicit prefill identity, bridge and run-directory paths")
        self.quant_matmul = None
        payload = envs.VLLM_ASCEND_QWEN_QUANT_MATMUL_PAYLOAD
        if payload:
            from vllm_ascend.ops.qwen_npu_ops.quant_matmul import QuantMatmulAdapter

            self.quant_matmul = QuantMatmulAdapter(payload)
            self.quant_matmul.install()
        super().load_model()
        if self.quant_matmul is not None:
            self.quant_matmul.prepare_model(self.model_runner.get_model())
        from vllm.config import CUDAGraphMode

        assert self.vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        assert get_pcp_group().world_size == 1
        self.prefill = GdnPrefill(
            envs.VLLM_ASCEND_QWEN_PREFILL_IDENTITY,
            envs.VLLM_ASCEND_QWEN_PREFILL_BRIDGE,
            ssm_layout=getattr(self, "ssm_layout", "VK"),
        )
        self.prefill.prepare(torch.device("npu", torch.npu.current_device()))
        self.vk_chain = VkGdnChain(self.prefill)
        self.prefill_enabled = envs.VLLM_ASCEND_QWEN_PREFILL_ENABLE
        self.prefill_calls = Counter()
        self.prefill_layers = set()
        self.prefill_shapes = Counter()
        self.prefill_layout = []
        self.graph_calls = Counter()
        sites = [
            (name, layer)
            for name, layer in self.model_runner.get_model().named_modules()
            if hasattr(layer, "conv1d") and hasattr(layer, "A_log") and hasattr(layer, "in_proj_ba")
        ]
        assert len(sites) == 48
        self.prefill_sites = {layer.prefix: layer for _, layer in sites}
        self.recurrent = None
        if getattr(self, "ssm_layout", "VK") == "VK" and envs.VLLM_ASCEND_QWEN_DECODE_PAYLOAD:
            self.recurrent = NativeVkRecurrent(envs.VLLM_ASCEND_QWEN_DECODE_PAYLOAD)
        for _, layer in sites:
            assert layer.num_spec in (7, 8) and layer.tp_size == 4
            if self.recurrent is not None:
                layer.qwen_recurrent = self.recurrent
            elif layer.num_spec == 8 and getattr(self, "ssm_layout", "VK") == "VK":
                raise ValueError("VK K8 requires the native Q9 recurrence payload")

        def core(
            qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor, output: torch.Tensor, prefix: str
        ) -> None:
            context = get_forward_context()
            layer = self.prefill_sites[prefix]
            metadata = context.attn_metadata
            meta = None if metadata is None else metadata[prefix]
            if meta is None:
                return
            supported = self.prefill.supports(layer, meta, qkv, z, b, a)
            mixed_supported = getattr(self, "ssm_layout", "VK") == "VK" and self.vk_chain.supports(
                layer, meta, qkv, z, b, a
            )
            if (supported or mixed_supported) and self.prefill_enabled:
                gdn_module.wait_for_kv_layer_from_connector(prefix)
                gdn_module.record_attention_compute_start()
                if mixed_supported:
                    count = self.vk_chain(layer, meta, qkv, z, b, a, output)
                    self.prefill_calls["r30_mixed_batches"] += 1
                    self.prefill_calls["native_mixed_spec_groups"] += int(meta.spec_sequence_masks is not None)
                    self.prefill_calls["native_mixed_decode_groups"] += int(meta.num_decodes > 0)
                else:
                    count = self.prefill(layer, meta, qkv, z, b, a, output, gdn_module._get_packed_conv_weights(layer))
                self.prefill_calls["r30_sequences"] += count
                self.prefill_calls["r30_batches"] += 1
                self.prefill_layers.add(prefix)
                self.prefill_shapes[int(meta.num_actual_tokens)] += 1
                gdn_module.maybe_save_kv_layer_to_connector("", [])
                return
            if getattr(self, "ssm_layout", "VK") == "KV":
                gdn_module.wait_for_kv_layer_from_connector(prefix)
                gdn_module.record_attention_compute_start()
                self.kv_chain(layer, meta, qkv, z, b, a, output)
                self.prefill_calls["kv_chain_batches"] += 1
                gdn_module.maybe_save_kv_layer_to_connector("", [])
                return
            self.prefill_calls["native_supported_prefill" if supported else "native_other"] += 1
            if meta.num_prefills:
                self.prefill_calls["native_prefill_batches"] += 1
            else:
                self.prefill_calls["native_decode_batches"] += 1
            temp = torch.zeros_like(z)
            layer._forward_core(qkv, b, a, temp)
            value = layer.norm(temp.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim)).view_as(z)
            output.copy_(value)

        def fake_core(qkv, z, b, a, output, prefix):
            return None

        # Keep dispatch outside traced forwards so Graph capture sees tensor
        # operations while route counters remain in the custom-op body.
        direct_register_custom_op(
            "qwen_r30_prefill_validation",
            core,
            mutates_args=["output"],
            fake_impl=fake_core,
            dispatch_key="PrivateUse1",
        )
        for _, module in sites:

            def forward(layer, hidden_states, output=None):
                if hasattr(layer, "in_proj_qkv"):
                    qkv, _ = layer.in_proj_qkv(hidden_states)
                    ba, _ = layer.in_proj_ba(hidden_states)
                    z, _ = layer.in_proj_z(hidden_states)
                    z = z.reshape(z.shape[0], -1, layer.head_v_dim)
                    b, a = layer._split_ba_for_tp(ba)
                elif not layer.gqa_interleaved_layout:
                    qkvz, _ = layer.in_proj_qkvz(hidden_states)
                    qkv, z = qkvz.split(
                        [(layer.key_dim * 2 + layer.value_dim) // layer.tp_size, layer.value_dim // layer.tp_size],
                        dim=-1,
                    )
                    z = z.reshape(z.shape[0], -1, layer.head_v_dim)
                    ba, _ = layer.in_proj_ba(hidden_states)
                    b, a = layer._split_ba_for_tp(ba)
                else:
                    qkvz, _ = layer.in_proj_qkvz(hidden_states)
                    ba, _ = layer.in_proj_ba(hidden_states)
                    qkv, z, b, a = gdn_module.fused_qkvzba_split_reshape_cat(
                        qkvz,
                        ba,
                        (layer.num_k_heads + layer.tp_size - 1) // layer.tp_size,
                        (layer.num_v_heads + layer.tp_size - 1) // layer.tp_size,
                        layer.head_k_dim,
                        layer.head_v_dim,
                    )
                value = torch.zeros_like(z)
                torch.ops.vllm.qwen_r30_prefill_validation(qkv, z, b.contiguous(), a.contiguous(), value, layer.prefix)
                result, _ = layer.out_proj(value.reshape(value.shape[0], -1))
                if output is not None:
                    output[: hidden_states.shape[0]] = result
                return result

            module.forward = types.MethodType(forward, module)
        self.observe_graph_dispatch()
        self.prefill_route_snapshot("loaded")

    def observe_graph_dispatch(self):
        from vllm.config import CUDAGraphMode

        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        original = ACLGraphWrapper.__call__

        def observed(wrapper, *args, **kwargs):
            context = get_forward_context()
            descriptor = context.batch_descriptor
            active = (
                context.cudagraph_runtime_mode != CUDAGraphMode.NONE
                and context.cudagraph_runtime_mode == wrapper.runtime_mode
            )
            entry = wrapper.concrete_aclgraph_entries.get(descriptor) if active else None
            mode = "replay" if entry is not None and entry.aclgraph is not None else "capture" if active else "eager"
            result = original(wrapper, *args, **kwargs)
            self.graph_calls[f"{type(wrapper.runnable).__name__}:{descriptor}:{mode}"] += 1
            return result

        ACLGraphWrapper.__call__ = observed

    def initialize_from_config(self, config):
        super().initialize_from_config(config)
        for prefix, layer in self.prefill_sites.items():
            conv, ssm = layer.kv_cache[:2]
            self.prefill_layout.append(
                dict(
                    prefix=prefix,
                    num_spec=layer.num_spec,
                    conv_shape=list(conv.shape),
                    conv_stride=list(conv.stride()),
                    conv_dtype=str(conv.dtype),
                    ssm_shape=list(ssm.shape),
                    ssm_stride=list(ssm.stride()),
                    ssm_dtype=str(ssm.dtype),
                )
            )

    def prefill_route_snapshot(self, label="snapshot"):
        result = dict(
            rank=self.rank,
            enabled=self.prefill_enabled,
            calls=dict(self.prefill_calls),
            layers=sorted(self.prefill_layers),
            token_shapes=dict(self.prefill_shapes),
            prefill=self.prefill.status(),
            graph_calls=dict(self.graph_calls),
            quant_matmul=None if self.quant_matmul is None else self.quant_matmul.snapshot(),
            native_recurrent=None if self.recurrent is None else self.recurrent.identity,
            layout=self.prefill_layout,
            scope="R30 pure/mixed/ragged prefill; native decode and RoPE retained",
        )
        if getattr(self, "ssm_layout", "VK") == "KV":
            result["scope"] = "persistent KV: R30 Prefill, R22 K7/K8 Decode and R01 RoPE"
            result["ssm_layout"] = "KV"
            result["kv_chain_calls"] = dict(self.kv_chain.calls) if hasattr(self, "kv_chain") else {}
            result["rope"] = self.rope.snapshot() if hasattr(self, "rope") else None
        folder = Path(envs.VLLM_ASCEND_QWEN_OPS_RUN_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"prefill-route-{label}-rank{self.rank}.json").open("x") as stream:
            json.dump(result, stream, indent=2)
        return result
