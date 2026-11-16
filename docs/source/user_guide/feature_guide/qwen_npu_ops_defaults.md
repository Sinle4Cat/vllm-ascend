# Qwen NPU Ops default profile

The explicit Qwen K7 launcher in
[qwen-npu-ops-integration](https://gitcode.com/sinle4cat/qwen-npu-ops-integration)
enables Prepare/H/O Prefill, registered ADN Decode, R01 RoPE, and default
QuantMatmul together. General framework defaults remain usable without private
artifacts; select the Qwen worker through the launcher to use this profile.

This profile targets Qwen3.8-27B W8A8, TP4, DSpark K7/Q8, BF16 activations and
FP32 persistent KV-oriented SSM state on Ascend910B. It uses up to eight model
sequences and 13 GiB of KV cache per card. HF32 Matmul and Conv are disabled.
ADN's operator supports B1-32/Q8; this is not a K8/Q9 or MTP3 profile.

Build matching artifacts from qwen_npu_ops commit
`b449216f965d11ae4d88833a2bf921114d3dd64e` and follow the integration repository's
README. The launcher validates operator manifests and Python identities before
starting vLLM. Its required artifact paths are:

- `PREFILL_IDENTITY`: Chunk GDN identity with independent Prepare, H and O entries.
- `ADN_IDENTITY`: newly built registered split-input ADN identity with current source.
- `ROPE_PAYLOAD`: matching R01 payload; it is used only for RoPE.
- `QUANT_PAYLOAD`: default QuantMatmul payload, without experimental wonly changes.

Three-stage Prefill uses native Conv, gating, Q/K normalization and output norm.
It batches real ragged sequence boundaries and keeps initial/final SSM state in
FP32. Decode receives the persistent KV bank, real token starts and the full page
table directly, including mixed batches. It performs no Decode state
gather/scatter or layout transposes.

RoPE and QuantMatmul retain their support predicates and use the official path
for unsupported inputs. All four routes are recorded by
`prefill_route_snapshot`, including actual registered providers and Graph calls.
An explicit R30 identity still selects the legacy R30 adapter. The default K7
launcher requires a three-stage identity, preventing accidental R30 selection.

Registered ADN is selected from the latest supported operator path. Historical
kernel timing is not a guarantee of a uniform end-to-end improvement; verify
the selected workload and speculative acceptance after changing providers.
