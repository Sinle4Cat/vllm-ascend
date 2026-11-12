# Experimental Qwen NPU Ops integration

完整构建、配置和 K3/K7/K8 启动步骤见 [Qwen NPU Ops 双仓使用指南](USAGE_zh.md)。

This branch packages the R30 prefill, native VK decode, compact R22 decode with
R01 RoPE, and W8A8 decode QuantMatmul adapters
as opt-in vLLM-Ascend workers. The normal `NPUWorker` and native GDN path are
unchanged. No custom kernel source, model weights, datasets, machine paths or
credentials are embedded in this integration.

## Versions and scope

- Framework base: official `v0.30.0rc2`, commit
  `8fde6d4e335fe300e2b949aab0e31659cce0811d`.
- Operator source: [Qwen NPU Ops](https://gitcode.com/sinle4cat/qwen_npu_ops),
  commit `04ffdbf013d1326f2384aee54737290c40bbc5bf`.
- Model geometry: Qwen3.6-27B, TP4/PCP1, BF16 activations, FP32 SSM,
  48 GDN layers, K7/Q8 or K8/Q9, `FULL_DECODE_ONLY`.
- Historical validation environment: Ascend910B4, CANN9.2, torch2.10 and
  torch_npu2.10.0.post7. Rebuild native payloads for their actual target.
- Historical vLLM source: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
  Its Qwen model and DSpark checkpoint/graph compatibility changes remain
  prerequisites for reproducing that deployment; this adapter does not
  rewrite the upstream model or draft checkpoint.

This branch adds MTP3/Q4 support to the current operator/framework candidate.
Its bounded Q4 operator and native-prefill model qualification is described
below; the historical operator commit and validation above do not include Q4.

The R30/R22 adapter bodies originate from the operator repository's
`integrations/vllm_ascend`. This branch places them in framework namespaces,
uses relative imports and centralizes its configuration in `envs.py`.
The KV worker uses R01 for compatible RoPE calls and the installed fused
RoPE for other inputs. With the QuantMatmul payload configured, supported
decode GEMMs use the custom library, and other inputs use the installed
QuantMatmul. The projections and scheduler retain their framework paths.

The operator repository's W-only preload remains an optional experiment.
This integration does not apply its patch or select a W-only payload.
Choose an R30 OPP built from the operator repository's unchanged `default`.

## Install the explicit payloads

Install this framework branch following the normal project installation
instructions. Install the matching Qwen NPU Ops API or decode payload wheel.
Run `scripts/build_all.py` from the operator repository to build both decode
payloads and the common prefill and QuantMatmul payloads. Source the generated
`activate.sh` and pass `--worker-cls "$QWEN_NPU_OPS_WORKER_CLS"` to serve.
The activation defaults to Native VK. Set `QWEN_NPU_OPS_LAYOUT=kv` when
sourcing the activation file to select the KV worker and R22/R01 payload.
Build and validate them against the actual CANN/torch_npu environment.
The standalone decode wheel does not automatically bundle or register R30.

Supply an R30 runtime identity JSON for the private OPP installation. Its
`vendor` is an absolute private vendor directory; `payload` maps relative
payload filenames to objects containing their `sha256` value;
`payload_identity_sha256` identifies the complete payload. Use the exact
identity retained by the operator build/validation, not a different OPP's
report. The bridge must expose the dual-layout R30 ABI from the pinned source.

Register the private R30 provider in `ASCEND_CUSTOM_OPP_PATH` **before**
starting Python. Importing torch_npu/vLLM before this can freeze discovery of
the wrong tiling provider. The adapter verifies recorded files and checks
the resolved ACLNN provider during startup.

## Optional Persistent KV route

Set these paths to the matching local artifacts and use a new output
directory for each service attempt:

```bash
export VLLM_ASCEND_QWEN_PREFILL_ENABLE=1
export VLLM_ASCEND_QWEN_PREFILL_IDENTITY=/path/to/r30-runtime-identity.json
export VLLM_ASCEND_QWEN_PREFILL_BRIDGE=/path/to/qwen_gdn_prefill_r30_bridge.so
export VLLM_ASCEND_QWEN_DECODE_PAYLOAD=/path/to/compact-r22-payload
export VLLM_ASCEND_QWEN_QUANT_MATMUL_PAYLOAD=/path/to/quant-matmul-payload
export VLLM_ASCEND_QWEN_OPS_RUN_DIR=/path/to/new-run/integration
export ASCEND_CUSTOM_OPP_PATH="/path/to/private-opp/vendors/qwen_npu_ops_transformer:/path/to/private-opp${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0

# Set MAIN_MODEL and DRAFT_MODEL to the compatible local checkpoints.
vllm serve "$MAIN_MODEL" \
  --tensor-parallel-size 4 --dtype bfloat16 --quantization ascend \
  --mamba-ssm-cache-dtype float32 --max-num-seqs 8 \
  --max-model-len 32768 --max-num-batched-tokens 8320 \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --worker-cls vllm_ascend.worker.qwen_npu_ops.kv_worker.QwenKvWorker \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"model\":\"$DRAFT_MODEL\"}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[64]}'
```

For K8, use `num_speculative_tokens:8` and the matching capture size 72 for
eight sequences. The decode payload must include the actual batch/query
buckets. Unbuilt buckets fail explicitly. Start with an empty cache;
the worker rejects KV-transfer connectors because their layout contract is
not qualified. Do not attach an old VK state snapshot to a KV worker.

For K3, use `num_speculative_tokens:3` and capture size32 at concurrency8.
The KV payload must include Q4 and uses six Conv history rows. Native VK stays
the default; its new manifest must explicitly declare Q4 in supported_queries.
Legacy manifests keep their original Q1/Q8/Q9 capabilities and reject Q4.
Q4 KV/VK B1..32 full/compact-ragged operators and state checks passed, with
Graph coverage at B1/B4/B8/B32. The qualified K3 model configuration uses
default VK, new Q4 recurrence and unchanged QuantMatmul, with native prefill:

```bash
QWEN_NPU_OPS_LAYOUT=vk source "$ARTIFACTS/activate.sh"
export VLLM_ASCEND_QWEN_PREFILL_ENABLE=0
```

Keep the generated identity/bridge/payload paths. On Qwen3.8-27B-W8A8 TP4/PCP1,
BF16/FP32, C8, graph32, seed1024,13GiB and max output24000, frozen GSM8K64
scored61/64 with all64 token/text outputs exactly matching the fresh official
K3 baseline. The R30-on configuration scored60/64 versus61/64 and retains its
NoLower FAIL; the toggle result is not a claim that R30 was fixed. KV model
qualification is not inferred from the VK score; only KV operator qualification
is claimed here. Historical K7/K8 scores retain their original scope. No new
speedup or full1319 result is claimed. R01 RoPE stays M1..72: M32 is covered,
M128 uses existing fallback, and default VK retains native RoPE.

The KV worker changes allocation before cache initialization and keeps SSM
state `[slot,head,K,V]` throughout. Pure or mixed prefill calls R30 per real
sequence. Speculative, ragged and ordinary decode groups call R22 with
`T=sum(lengths)` rows, original query starts, accepted counts and the complete
`[batch,query]` page table. Sequences are not padded to equal token lengths.
Null page 0 and inactive IDs are not written. The VK native core is rejected
for a KV bank. Decode workspaces and bridge buffers are prepared before
Graph capture and included in memory profiling.

## Default Persistent VK route

Select `vllm_ascend.worker.qwen_npu_ops.prefill_worker.QwenPrefillWorker`
to use R30 pure/mixed prefill with persistent VK state. RoPE remains on the
installed framework path. The native VK recurrence payload preserves
the original AscendC arithmetic while fixing compact-query/history indexing,
accepted-history bounds and Q9 support. Build it in the operator repository:

```bash
python scripts/build_native_recurrent.py --output /path/to/new-native-vk-payload
export VLLM_ASCEND_QWEN_DECODE_PAYLOAD=/path/to/new-native-vk-payload
```

Use the same R30 identity, bridge and private provider configuration above,
but select `QwenPrefillWorker`. The recurrence payload is specialized for
Ascend910B4, TP4 Qwen heads and BF16 activations with FP32 VK state; rebuild
for the actual CANN/torch_npu runtime. It replaces recurrence per layer in
both pure decode and mixed batches, preserving device-side real lengths,
accepted counts and the complete page table. No equal-length query padding,
full-state transpose/copy or recurrence HBM workspace is required. R30 keeps
its reusable prefill workspace.

Without the explicit payload, K7 retains the installed native recurrence;
K8 fails during worker setup because the old `MAX_MTP=8` kernel cannot
handle Q9. The separately packaged R22/KV payload remains a different route.

For a matched VK wrapper baseline, set `VLLM_ASCEND_QWEN_PREFILL_ENABLE=0`
with the same prefill worker and native recurrence payload. This reference
includes the same decode repair and is not unmodified official K8. The KV worker rejects disabling
R30 because a native VK fallback cannot consume its state allocation.

## Validation boundaries

Run the CPU adapter tests without torch or an NPU:

```bash
python -m unittest discover -s tests/ut/ops -p test_qwen_npu_ops_adapter.py -v
```

Run the state-bridge regression on an authorized, idle NPU:

```bash
pytest -q tests/e2e/pull_request/one_card/ops/test_qwen_npu_ops_state_bridge.py
```

The latter checks KV/VK orientation, K3/K7/K8 page widths, padded bank strides,
inactive/null pages, eager execution and captured Graph replay. It does not
require model weights or a custom GDN payload. Operator accuracy and
prefill/decode handoff validators remain in the operator repository.

Publication validation passed 7 CPU adapter tests, 8 NPU state-bridge tests
and 90 compact/mixed NPU cases with the relocated framework modules.
The NPU checks used Ascend910B4, CANN9.2 and torch_npu2.10.0.post7 with the
retained R30/R22 payloads. They verify adapter relocation and state handling;
they do not qualify a new operator build or a complete model deployment.

Historical component evidence includes 152 R30 cases, 76 canonical KV/VK
pairs, 90 compact/mixed cases and 32 state handoffs. The later full GSM8K
KV campaign scored 1280/1319 for K7 and 1285/1319 for K8 after the tail fix.
Those results belong to their frozen operator/framework/runtime artifacts,
not to a fresh installation merely because its imports or unit tests pass.
No new model accuracy or TTFT/TPOT gain is claimed by publishing this adapter.

The route snapshot records per-rank providers, SSM layout, layer hits and
dispatch counts under the explicit run directory. Python custom-op counters
observe eager/capture calls; Graph replay does not execute those Python
counters again.

## Latest VK recurrence repair validation

The VK worker now supports mixed prefill/ordinary/speculative decode with
compact real query lengths and full Q8/Q9 history tables. The explicit
native recurrence build has the same binary SHA256 as the frozen 20002 model
validation payload. It uses no recurrence HBM workspace; R30 prefill still
retains its own reusable workspace. Default R30 and optional W-only sources
were not changed.

The publication candidate passed 8 CPU framework contracts and 84 NPU tests:
22 independent native recurrence cases, 54 mixed VK cases and 8 KV/VK state
bridge cases. Reproduce the targeted NPU checks with the commands in the
Chinese usage guide. They include inactive/zero-length queries, B8, complete
state and padding preservation, Q8 native parity and changed Graph metadata.

The associated 20002 model run used a 64-question GSM8K subset. Native/R30
scored 60/61 for K7 and 62/61 for K8. Both arms include the same decode repair.
Those VK subset scores do not inherit the earlier KV full-dataset results
and do not establish full-dataset equivalence.
