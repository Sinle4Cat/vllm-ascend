# Experimental xLLM MegaGDN integration

This integration is opt-in. The default `VLLM_ASCEND_GDN_BACKEND=baseline`
retains the native GDN implementation. The fused paths are **not accuracy
accepted** and must not be treated as production-ready or as evidence of a
publishable performance improvement.

## Backends

| Backend | Eligible fused path | Other invocations |
| --- | --- | --- |
| `baseline` | None | Native GDN |
| `xllm_ops` | Pure prefill | Native core and gated RMSNorm |
| `xllm_ops_prefill_decode` | Pure prefill, eligible unpadded batch-one decode, eligible batch-one MTP verification | Native core and gated RMSNorm |
| `ascendc_phase6` | Phase6 prefill core | Existing fallback |

Mixed prefill/decode batches, padded batches, unsupported cache layouts, and
multi-request MTP verification do not enter the fused decode path. In particular,
concurrency four does **not** imply fused batch-four MTP decode. The tested MTP=3
route uses batch one with four verification tokens.

## Dependencies and state contract

The tested runtime uses vLLM source commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, Torch 2.10.0,
Torch-NPU 2.10.0.post4, and compatible xllm-ops Main commit
`fee381637f072a4b3a3d3e320aad327b17e27738`.
The independently built `custom_ops_lib` and its matching CANN vendor package
must expose `mega_gdn_prefill_op`, `mega_gdn_decode`, and
`mega_gdn_mtp_decode`. Those binary packages are not included in this repository.
Preserve native custom-operator registration precedence: the native tensor-index
convolution ABI is not interchangeable with the xLLM array-index ABI.

Prefill reserves the SSM scratch checkpoints required by the convolution history
and copies only the final native state back. Decode uses native FP32 `[V,K]`
states. MTP gathers non-adjacent native checkpoint IDs into a compact bank, calls
the dedicated MTP operator, and scatters state back on device. Padding/no-spec
replay must not write native state, and NULL slot zero remains untouched.

The ordinary decode operator's direct cache-bank limit is 1024 slots. Larger or
strided banks use compact scratch; the tested serving configuration keeps
`--num-gpu-blocks-override 1024`. The adapter does not change the allocator.

## Validation boundary

CPU routing/cache-contract tests and NPU graph replay checks are separate from
numerical and model accuracy acceptance. Prior MTP testing against native GDN
failed the SSM numerical comparison in 24/24 cases without loosening tolerances.
Differences in gate precision and Q/K normalization are investigation leads,
not an established model-level root cause. CEval/model accuracy remains open.

The TP4 graph-mode, MTP=3, W8A8 request comparison is diagnostic only:
its fairness verdict is INCONCLUSIVE. Matching 20 short outputs for one prompt
does not establish model accuracy or pure-decode performance. No before/after
speedup is accepted by this document.

Publication checks passed 22 focused CPU contract tests, Ruff lint/format,
Python compilation, and AST equivalence for all seven Python files against the
tested container source. NPU numerical tests were not rerun for publication.
Full `format.sh ci` could not run because `pre-commit` was unavailable; these
focused checks do not establish full CI or model accuracy acceptance.

Run the focused CPU contract tests with the matching runtime:

```bash
VLLM_ASCEND_GDN_BACKEND=baseline python -m pytest -q \
  tests/ut/ops/test_gdn_layerwise_kv.py \
  tests/ut/ops/test_gdn_xllm_decode.py \
  tests/ut/patch/worker/test_patch_qwen3_5_mtp.py
```

Native numerical and graph-replay tests for ordinary decode are in
`tests/e2e/pull_request/one_card/ops/test_xllm_gdn_decode.py` and require an
isolated NPU plus the matching binding/vendor package. Passing CPU mocks or
replaying the fused path against itself does not replace native parity testing.

Disabling scheduler-level chunked prefill was discussed but not implemented or
validated. It is not part of this code change.
