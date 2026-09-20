# 原生 GDN 混合批次支持

`VLLM_ASCEND_GDN_BACKEND=xllm_ops_kv` 支持同一批次中的 prefill、普通 decode 和 MTP verification。使用现有 native prefill/decode SDK 接口，不新增算子 ABI。算子 SDK 分支为 `perf/gdn-native-mtp-20260919`，框架接入分支为 `experiment/gdn-native-sdk-model-20260919`。

## 调度与状态

`_use_xllm_mixed` 检查阶段计数、token 总数、设备索引和 prefill metadata；`_forward_core` 再检查实际头维度、数据类型、卷积历史宽度及归一化条件。支持范围外继续使用框架 fallback。

`native_mixed_kv` 依据 builder 的 `spec_token_indx` / `non_spec_token_indx` 取得各阶段输入。MTP verification 直接使用设备 query starts、accepted 长度和 checkpoint 索引；普通 decode 使用 non-spec 前缀；prefill 使用已经重定位的 prefill starts。输出按原 token 索引合并。各阶段使用分配器原始 conv / SSM bank，无状态池 gather 或整池复制，也不读取设备 metadata 到 CPU。

当前 builder 会把与 MTP 混合的普通 decode 折入 prefill。适配器也支持三个阶段同时存在，测试包含这种 metadata。目标模型使用 `FULL_DECODE_ONLY`：混合 prefill 在 eager 路径执行，纯 decode 保留图重放。

当前验证配置为 Ascend 910B3、CANN 9.1、Qwen3.6-27B W8A8、TP4/MTP3、FP32 SSM。混合入口要求 PCP=1、head dimension=128、BF16 输入和 conv bank、FP32 SSM、卷积宽度4、SiLU gate 以及模型对应的 RMSNorm 参数。MTP 单阶段最多32个请求，checkpoint 宽度2–17。未覆盖的硬件或模型不据此宣称支持。

## SDK 缓存修复依赖

`mega_chunk_gdn.cpp` 的 KKT/Solve 流水不能让 KK cache 与 Solve 的 `A_inv` 输出共用同一片工作区。当前实现把 KK cache 放到 Solve 后才开始使用的 `S` 工作区；`D<C` 时关闭该流水并使用分阶段同步，避免工作区容量不足。此轮 NPU 验证覆盖当前模型 D=C=128。

## 回归测试

- `tests/ut/ops/test_gdn_kv_backend.py` 新增混合路由选择和缺失/不一致 metadata 拒绝检查。与 builder、decode、layerwise、chunk metadata 回归共94项通过。
- `tests/e2e/pull_request/one_card/ops/test_gdn_native_mixed.py` 12项真实 NPU 测试，覆盖 MTP+prefill、普通 decode+prefill、三阶段混合；不等长验证、padding、非连续 checkpoint、变化的 accepted 长度、初始/续接状态、跨128-token边界、strided输入与状态池。
- 混合输出和状态与独立打包的各阶段 SDK 调用逐位一致；检查未选中槽位、slot0、conv尾部和 stride间隙未被修改。这些是路由/状态一致性测试，数值 oracle 沿用 SDK 独立测试。
- 与现有 `test_gdn_kv.py` 一起，真实 NPU 回归33项通过，其中包含纯 decode 的图重放测试。SDK 当前模型 prefill kernel 从产品分支重建后与前一轮44项 SDK 测试所用二进制 SHA256 相同。

模型路由覆盖、性能与精度结果见本次 `mixed-support-001/REPORT.md`。单元和算子测试通过不能代替模型精度验收。

## 本次混合批次模型复测（B2）

本轮是历史同配置 A1/B1/A2 之后的一个新候选 B2，属于对照复测，不是新一轮 A/B/A。两条长输入 route 的 9/9 prefill steps、每条 141076/141076 prefill tokens 均进入 native；两条 route 各记录 6 次 `native_mixed_kv` 调用。CEval50 为 41/50；相对旧 B1 有 5 道预测差异，严格 precision 仍未通过。

长输入 `full-original` 的 B2 TTFT 为 11541.897 ms，相对 B1 仅下降 2.785%，相对 A1/A2 下降 3.536%/3.390%，未达到 15% 的目标。B2 短 decode128 TTFT/TPOT/吞吐为 320.496 ms/9.850 ms/292.830 tokens/s；长输入 TPOT/吞吐为 556.251 ms/1.410 tokens/s。控制器最终 cleanup PASS，证据与逐请求输出 ID 保存在 `mixed-support-001/`。
