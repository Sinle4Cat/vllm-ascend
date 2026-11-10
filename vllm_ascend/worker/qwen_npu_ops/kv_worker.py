"""Explicit qwen-only persistent KV worker for vLLM-Ascend 0.30 TP4.

Use a fresh empty cache. VK snapshots and cross-layout KV connectors are not
compatible with this worker; the allocation and every GDN consumer use KV.
"""

import types

import torch
from qwen_npu_ops import GdnDecode

from vllm_ascend import envs
from vllm_ascend.ops.qwen_npu_ops.kv_chain import KvGdnChain
from vllm_ascend.worker.qwen_npu_ops.prefill_worker import QwenPrefillWorker


class QwenKvWorker(QwenPrefillWorker):
    def load_model(self):
        if not envs.VLLM_ASCEND_QWEN_PREFILL_ENABLE or not envs.VLLM_ASCEND_QWEN_DECODE_PAYLOAD:
            raise ValueError("The KV worker requires enabled R30 prefill and an explicit compact R22 payload")
        self.ssm_layout = "KV"
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("Persistent KV requires a layout-aware connector; none is qualified here")
        super().load_model()
        from vllm_ascend.ops.qwen_npu_ops.rope import QwenRopeAdapter

        payload = envs.VLLM_ASCEND_QWEN_DECODE_PAYLOAD
        self.kv_chain = KvGdnChain(GdnDecode(payload, allow_experimental=True), self.prefill)
        self.rope = QwenRopeAdapter(payload, self.model_runner.get_model())
        for layer in self.prefill_sites.values():
            vk_shapes = layer.get_state_shape

            def shape(current, original=vk_shapes):
                shapes = list(original())
                ssm = shapes[1]
                shapes[1] = (*ssm[:-2], ssm[-1], ssm[-2])
                return tuple(shapes)

            layer.get_state_shape = types.MethodType(shape, layer)
            layer._qwen_ssm_layout = "KV"

            def reject_vk_core(current, *args, **kwargs):
                raise RuntimeError("A VK _forward_core cannot consume a persistent KV state bank")

            layer._forward_core = types.MethodType(reject_vk_core, layer)
        # Account for retained decode IO in the runner's memory profile,
        # before it sizes the persistent cache from remaining HBM.
        example = next(iter(self.prefill_sites.values()))
        device = example.A_log.device
        self.kv_chain.prepare(
            torch.empty((1, example.num_spec + 3, 2560), dtype=torch.bfloat16, device=device),
            torch.empty((1, 12, 128, 128), dtype=torch.float32, device=device),
            max_batch=self.vllm_config.scheduler_config.max_num_seqs,
        )

    def initialize_from_config(self, config):
        super().initialize_from_config(config)
        for layer in self.prefill_sites.values():
            self.kv_chain.prepare(*layer.kv_cache[:2], max_batch=self.vllm_config.scheduler_config.max_num_seqs)
        for row in self.prefill_layout:
            row["ssm_layout"] = "KV"
