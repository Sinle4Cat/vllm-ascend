# Qwen NPU Ops 双仓使用指南

统一部署默认启用 R30 GDN Prefill、Native VK Decode 和 W8A8 decode QuantMatmul，RoPE 使用原生实现。
可显式切换到 R22 GDN Decode + R01 RoPE 的 KV 路径。

新增 MTP=3 对应 Q4，C8 的 decode Graph capture size 为 32，Conv history 为 6 行。Native VK manifest 必须明确声明 `supported_queries` 包含 Q4；旧 manifest 继续仅开放 Q1/Q8/Q9，不会被自动视为 Q4 payload。默认 KV 构建包含 Q4/Q8/Q9 的 B1–32 加 RoPE，共 97 个对象。本能力尚待目标环境的算子与模型验收，历史 K7/K8 结果不能作为 Q4 验收。R01 的 M1–72 范围不变，Q4/C8 的 M32 可命中，M128 仍走已有 native fallback；默认 VK 继续使用原生 RoPE。
算子源码放在 [qwen_npu_ops](https://gitcode.com/sinle4cat/qwen_npu_ops)，框架接入放在
[vllm-ascend 的 qwen-npu-ops-v0.30 分支](https://github.com/Sinle4Cat/vllm-ascend/tree/qwen-npu-ops-v0.30)。

历史 R30/R22/VK 接入的最低功能提交如下。统一构建和 QuantMatmul 使用表中分支的当前版本。

| 仓库 | 分支 | 功能提交 |
| --- | --- | --- |
| vllm-ascend | qwen-npu-ops-v0.30 | [d41239245](https://github.com/Sinle4Cat/vllm-ascend/commit/d41239245b900e11478d95762cab0d1af2d155b7) |
| qwen_npu_ops | main | [04ffdbf](https://gitcode.com/sinle4cat/qwen_npu_ops/commit/04ffdbf013d1326f2384aee54737290c40bbc5bf) |

## 两个仓库分别做什么

| 仓库 | 职责 | 需要生成的产物 |
| --- | --- | --- |
| qwen_npu_ops | R30、Native VK、R22/R01 和 QuantMatmul 源码、编译脚本、Python API | 私有 R30 OPP 和 bridge、VK 与 KV decode payload、QuantMatmul payload |
| vllm-ascend | worker、状态布局、混合批次路由、Graph 接入与回归测试 | 安装本分支；通过 worker 和环境变量显式启用 |

默认 `NPUWorker` 不会因为安装算子包而自动启用这些接入。W-only 是独立可选实验，本指南不应用其 patch，也不替换 R30 `default`。

## 已验证的环境与前提

- Linux aarch64、Ascend910B4，TP4/PCP1，BF16 激活、FP32 SSM。
- 构建环境需要 CMake、Ninja、C++ 编译器和 CANN `bisheng`。
- 驱动 26.1.1、CANN 9.2、torch 2.10、torch_npu 2.10.0.post7。
- vLLM-Ascend 基于 `v0.30.0rc2`；历史 vLLM core 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
- 主模型满足 Qwen TP4 分片几何：Q/K heads=4、value heads=12、K=V=128、48 个 GDN 层。
- K7 使用 Q8，K8 使用 Q9；本指南使用 `FULL_DECODE_ONLY` 和旧 model runner。

先准备含兼容 Qwen 模型、DSpark checkpoint/Graph 支持、`fla_npu.ops.ascendc`、CANN 和 ATB 的 vLLM 环境。
这两个仓库不包含主模型/draft 权重，也不替换 vLLM core；仅安装 stock vLLM 不能据此认定具备历史部署的全部模型兼容修改。
主模型、draft 的词表、tokenizer、模型接口及 draft head 必须匹配。
换 CANN、torch_npu 或设备后，在目标环境重新编译 payload；不要复制另一运行时的 `.so`。

## 获取源码并安装框架

以下命令在目标 Linux 环境执行。路径均为使用者自己的目录。

```bash
export WORK_ROOT=/data/qwen-deploy
mkdir -p "$WORK_ROOT"
git -c core.autocrlf=false clone \
  https://gitcode.com/sinle4cat/qwen_npu_ops.git "$WORK_ROOT/qwen_npu_ops"
git -c core.autocrlf=false clone --branch qwen-npu-ops-v0.30 \
  --recurse-submodules https://github.com/Sinle4Cat/vllm-ascend.git \
  "$WORK_ROOT/vllm-ascend"

export OPS="$WORK_ROOT/qwen_npu_ops"
export FRAMEWORK="$WORK_ROOT/vllm-ascend"
export ARTIFACTS="$WORK_ROOT/artifacts-r001"
git -C "$FRAMEWORK" submodule update --init --recursive
git -C "$FRAMEWORK" submodule status
git -C "$OPS" rev-parse HEAD
git -C "$FRAMEWORK" rev-parse HEAD

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python3 -m pip install --no-build-isolation --no-deps -e "$OPS"
python3 -m pip install --no-build-isolation --no-deps -e "$FRAMEWORK"
```

记录两仓 commit 和环境版本。框架 Catlass 固定在 submodule commit `41bf90da655bba3c66d0acd7e00abe33960ecfd6`；
只复制框架目录而不初始化该依赖会导致构建缺少头文件。
`--no-deps --no-build-isolation` 保留事先配置的匹配运行时；依赖需要由前置环境准备齐全。
算子 Python 包安装成功不代表已包含或编译了以下二进制产物。

## 一次构建全部算子

```bash
cd "$OPS"
python3 scripts/build_all.py --output "$ARTIFACTS" --jobs 8
source "$ARTIFACTS/activate.sh"
```

`ARTIFACTS` 使用新的绝对路径。脚本构建并安装私有 R30 OPP，同时产出 bridge、Native VK、
R22 K7/K8 B1–32、R01 RoPE 和包含全部 45 个支持 shape 的 QuantMatmul。
激活文件默认将 `VLLM_ASCEND_QWEN_DECODE_PAYLOAD` 指向 `native-vk`，
并将 `QWEN_NPU_OPS_WORKER_CLS` 设为 `QwenPrefillWorker` 的完整类名。
如需 R22/R01，显式选择 KV。

```bash
QWEN_NPU_OPS_LAYOUT=kv source "$ARTIFACTS/activate.sh"
```

设置 `VLLM_ASCEND_QWEN_PREFILL_ENABLE=1` 启用 R30。KV worker 使用同一 decode payload 中的 R22 和
R01，R01 支持 1–72 行及匹配的布局，超出支持范围走原生 RoPE。
激活文件中的 `VLLM_ASCEND_QWEN_QUANT_MATMUL_PAYLOAD` 在符合条件的 W8A8
decode 调用中启用 QuantMatmul。其余 shape、prefill 和混合批次走框架原生 QuantMatmul。
清空该变量即可进行官方 QuantMatmul 对照。启用后可以在 `prefill_route_snapshot` 核对库身份和图捕获命中。

## 分项构建 R30 prefill

```bash
cd "$OPS"
bash scripts/build_gdn_prefill.sh --soc ascend910b --jobs 8 \
  --output "$ARTIFACTS/r30-package" --keep-build
find "$ARTIFACTS/r30-package" -type f -name '*.run'
```

从构建输出选择生成的 `fla_npu_linux-aarch64.run`，并保留最后打印的 staged build tree。
下面两个变量需要填入这次构建的实际路径。

```bash
export PREFILL_PACKAGE=/absolute/path/to/fla_npu_linux-aarch64.run
export STAGED_SOURCE=/absolute/path/to/qwen-npu-ops-gdn-prefill.XXXXXXXX
export OPP_ROOT="$ARTIFACTS/r30-opp"
bash "$PREFILL_PACKAGE" --install-path="$OPP_ROOT"

# 安装器通常创建下列 vendor；以本次安装的实际目录为准。
export VENDOR="$OPP_ROOT/vendors/qwen_npu_ops_transformer"
test -f "$VENDOR/op_api/include/aclnnop/aclnn_mega_gdn_prefill_pr990.h"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR:$OPP_ROOT${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"

python3 scripts/check_gdn_prefill_build.py --staged-tree "$STAGED_SOURCE" \
  --vendor "$VENDOR" --output "$ARTIFACTS/r30-provenance.json"
python3 scripts/create_gdn_prefill_runtime_identity.py \
  --vendor "$VENDOR" --provenance "$ARTIFACTS/r30-provenance.json" \
  --output "$ARTIFACTS/r30-runtime-identity.json"
python3 scripts/build_gdn_prefill_bridge.py \
  --header "$VENDOR/op_api/include/aclnnop/aclnn_mega_gdn_prefill_pr990.h" \
  --output "$ARTIFACTS/r30-bridge"
```

身份生成命令核对 PASS 构建报告、安装文件的 SHA256 和完整 inventory。
它生成框架需要的 `vendor`、`payload`、`payload_identity_sha256`，不代替算子精度测试。
构建和验证输出目录应使用新路径；脚本拒绝覆盖现有结果。
必须在服务 Python 进程导入 torch_npu/vLLM **之前**配置私有 OPP，不能在模型加载后才切换 provider。

## 分项构建 decode

| 路径 | worker | decode payload | 状态合同 |
| --- | --- | --- | --- |
| 默认 VK | `vllm_ascend.worker.qwen_npu_ops.prefill_worker.QwenPrefillWorker` | native VK，ABI `qwen-native-vk-recurrent-v1` | 持久 `[slot,head,V,K]`，原生 RoPE |
| 可选 KV | `vllm_ascend.worker.qwen_npu_ops.kv_worker.QwenKvWorker` | compact R22 + R01，ABI `qwen-decode-direct-v1` | 持久 `[slot,head,K,V]`，R30/R22 全链路 KV |

单独构建 VK payload 的命令如下。

```bash
cd "$OPS"
python3 scripts/build_native_recurrent.py --output "$ARTIFACTS/native-vk"
export VLLM_ASCEND_QWEN_DECODE_PAYLOAD="$ARTIFACTS/native-vk"
export QWEN_NPU_OPS_WORKER_CLS=vllm_ascend.worker.qwen_npu_ops.prefill_worker.QwenPrefillWorker
```

该 recurrence 使用真实长度、完整 `[B,Q]` 页表和上轮 accepted counts；无需把每条 query 补成等长，
不复制或转置完整持久 SSM bank，recurrence 额外 HBM workspace 为 0。R30 自己的 prefill workspace 仍会分配和复用。
R30 按声明的 KV/VK ABI 处理输入；VK prefill 的内部布局适配仍存在，不能把整条链路描述成完全无布局转换。
原版 `MAX_MTP=8` recurrence 不支持 K8/Q9，因此 VK K8 缺少此 payload 时会在 worker 启动阶段明确报错。

单独构建 R22 + R01 的命令如下。缺省构建 B1–32 的两个 Q 档。

```bash
cd "$OPS"
python3 scripts/build_decode.py --cann-version 9.2.0 \
  --queries 8 9 --output "$ARTIFACTS/r22-k7-k8"
export VLLM_ASCEND_QWEN_DECODE_PAYLOAD="$ARTIFACTS/r22-k7-k8"
export QWEN_NPU_OPS_WORKER_CLS=vllm_ascend.worker.qwen_npu_ops.kv_worker.QwenKvWorker
```

`--cann-version` 必须与实际 toolkit 一致。缺少实际 B/Q bucket 时不会隐式换用另一布局的 kernel。
两个布局的 payload 不可互换；切换 worker 必须重新启动并创建空缓存，不复用旧 SSM snapshot。
KV worker 要求启用 R30，且不支持未经验证的跨布局 KV transfer connector。

## 启动 K7 或 K8

下面用已激活的 worker 启动 K8、TP4、最多八并发。改 `K=7` 即选择 K7，capture size 分别为 72/64。

```bash
export MAIN_MODEL=/absolute/path/to/main-model
export DRAFT_MODEL=/absolute/path/to/draft-model
export RUN_DIR="$WORK_ROOT/runs/vk-k8-001"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/integration"
export VLLM_ASCEND_QWEN_PREFILL_ENABLE=1
export VLLM_ASCEND_QWEN_PREFILL_IDENTITY="$ARTIFACTS/r30-runtime-identity.json"
export VLLM_ASCEND_QWEN_PREFILL_BRIDGE="$ARTIFACTS/r30-bridge/qwen_gdn_prefill_r30_bridge.so"
export VLLM_ASCEND_QWEN_OPS_RUN_DIR="$RUN_DIR/integration"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_SERVER_DEV_MODE=1
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1 HCCL_DETERMINISTIC=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
K=8
CAPTURE=$((8 * (K + 1)))

vllm serve "$MAIN_MODEL" --served-model-name qwen --host 127.0.0.1 --port 49018 \
  --tensor-parallel-size 4 --data-parallel-size 1 --dtype bfloat16 \
  --quantization ascend --mamba-ssm-cache-dtype float32 --seed 1024 \
  --worker-cls "${QWEN_NPU_OPS_WORKER_CLS:?Source activate.sh or select a decode worker}" \
  --language-model-only --trust-remote-code \
  --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8320 \
  --kv-cache-memory-bytes 13958643712 --block-size 128 \
  --no-enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false}}' \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$K,\"model\":\"$DRAFT_MODEL\"}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$CAPTURE],\"pass_config\":{\"fuse_norm_quant\":false,\"fuse_act_quant\":true}}" \
  > "$RUN_DIR/server.log" 2>&1
```

`13958643712` 是本次用户选定的 **13 GiB/卡**，不是所有设备/负载通用的默认值。
历史 VK 路径的 13 GiB 容量测试中，K7/K8 各两轮共 50 请求通过，包括八并发满 32K；
token 容量分别为 388,918/370,688，最低采样空闲约 571/251 MiB。KV/R22 使用自己的 bridge 和
workspace，改变布局、并发、capture、batched tokens 或 payload 后应重新测容量。
显式 cache bytes 覆盖按 utilization 自动推算的预算；日志中的实际 block/capacity 才是分配结果。

## 确认模型和接入路径

在另一个已配置相同环境的终端执行：

```bash
curl -f http://127.0.0.1:49018/health
curl -sSf http://127.0.0.1:49018/v1/models
curl -sSf http://127.0.0.1:49018/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"top_p":1,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
curl -sSf http://127.0.0.1:49018/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"prefill_route_snapshot","args":["after-smoke"]}'
```

核对四 rank、48 个 GDN 层、实际 provider/payload 身份和 `r30_sequences`。
默认 VK 应报告 `native_recurrent.manifest.abi=qwen-native-vk-recurrent-v1`。
可选 KV 应报告 `ssm_layout=KV`，并检查 `kv_chain_calls` 的 R22 capture 和 `rope.capture_hits` 的 R01 命中。
两种布局均可检查 `quant_matmul.capture_hits` 的 W8A8 decode 命中。
Graph replay 不再执行 Python 计数器，因此计数不增长不能单独证明 replay 没走算子。
首次加载/图编译不计入热态 TTFT/TPOT；health、真实生成、计数和最终答案应分别检查。

## 回归测试与对照

```bash
cd "$OPS"
python3 -m unittest discover -s tests -p test_runtime_identity.py -v
bash scripts/validate_gdn_prefill.sh "$VENDOR" 0 "$ARTIFACTS/r30-accuracy"

cd "$FRAMEWORK"
python3 -m unittest discover -s tests/ut/ops -p test_qwen_npu_ops_adapter.py -v
VLLM_ASCEND_QWEN_DECODE_PAYLOAD="$ARTIFACTS/native-vk" \
  ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest -q \
  --confcutdir=tests/e2e/pull_request/one_card/ops \
  tests/e2e/pull_request/one_card/ops/test_qwen_npu_ops_native_recurrent.py \
  tests/e2e/pull_request/one_card/ops/test_qwen_npu_ops_vk_mixed.py \
  tests/e2e/pull_request/one_card/ops/test_qwen_npu_ops_state_bridge.py
```

上面的 NPU 回归使用 **VK native payload**，覆盖独立 CPU 输出/完整状态、Q1/Q8/Q9、B8、零长度、
不等长 query、未触碰页、padding、动态 Graph metadata 和混合批次。选择 KV/R22 时还需运行算子仓库
[持久 KV 验证](https://gitcode.com/sinle4cat/qwen_npu_ops/blob/main/docs/persistent-kv.md) 中的对应命令。
单算子/状态测试通过不等于模型全量精度通过。

做 VK prefill 的 A/B 时，保持同一 native payload、同卡、相同 cache bytes 和请求输入，分别设置
`VLLM_ASCEND_QWEN_PREFILL_ENABLE=0/1`，并使用新的 run 目录。
关闭 R30 的这一组仍包含相同 decode 修复，**不是未修改的官方 K8**。
完整原生 K0 应使用默认 `NPUWorker` 并关闭投机，不能用带修复的 K8 结果冒充。
TPOT 会受输出量、动态批次和投机接受率影响，需同时保留服务端 token/acceptance/error/abort/preemption 计数。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 构建缺少 Catlass 头文件 | 在框架仓库初始化固定 submodule；不要以空目录代替 |
| K8 提示缺少 native Q9 payload | VK worker 指向 `build_native_recurrent.py` 的产物，而非 R22 目录 |
| runtime identity mismatch | 核对实际 torch/torch_npu/架构，在目标环境重新构建 payload |
| provider 或 tiling 不匹配 | 核对私有 vendor，并在新的 Python 服务进程启动前设置 OPP 路径 |
| KV worker 禁止 native fallback | KV bank 不能交给 VK 原生 core；检查 R30/R22 配置与 B/Q buckets |
| 13 GiB 请求通过但空闲很少 | 分别看请求完成性和最低余量；本轮 K8 只剩约251MiB，不能据此宣称512MiB余量通过 |
