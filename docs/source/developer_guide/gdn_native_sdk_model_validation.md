# 原生 SDK GDN 模型接入测试

## 最新数值修复与复测


原生 decode / MTP 的两处数值边界已修复，并完成 TP4/MTP3 模型 A1/B1/A2 测量。
这轮结果仍属于调试性能结论：历史 801-token 偶发差异尚未定位，模型精度的严格逐题一致性尚未关闭。

## 修改与回归

- `g` 保留 FP32，移除旧 SDK MegaMTP 的 BF16 舍入；`beta` 保持 BF16。
- L2Norm 后的 Q/K 先舍入 BF16，再进行 FP32 recurrence，query scale 在舍入后应用。
- 新增四个独立数值边界测试；旧包 4/4 失败，新包通过。SDK 总计 44/44 通过。
- 框架测试改用真实 convolution、L2Norm、gating 和 recurrent 算子作为对照。旧包 4/4 失败，新包完整 21/21 通过，误差阈值未放宽。
- 测试接入阶段的累计长度格式和 Q/K 连续化错误已经修正；对应失败日志保留，不能把这些测试接法错误算作产品修复。
- 保留此前 prefill 空槽位无状态计算、单序列非法索引保护及 workspace 生命周期修复。slot0 预热负偏移写回是此前 MTP 接受率约 4.55% 的主因。

## 模型性能

Qwen3.6-27B W8A8，910B3，CANN 9.1，TP4、MTP3、并发4，FP32 SSM，prefix cache 关闭。
物理卡4–7、端口8826，每轮独立服务/图缓存。每场景预热5批后测5批，共20请求；启动、编译、capture不计时。
吞吐按测量批次 wall time 之和计算，排除批次间指标抓取与日志时间。三轮 payload SHA 完全一致，prefix cache 命中均为0。

| 场景 | 指标 | baseline A1 | native B1 | baseline A2 |
| --- | --- | ---: | ---: | ---: |
| decode128 | TTFT ms | 522.978 | 401.386 | 433.936 |
| decode128 | TPOT ms | 10.805 | 9.689 | 10.543 |
| decode128 | 输出 tokens/s | 244.934 | 282.323 | 261.836 |
| decode128 | MTP 接受率 % | 80.212 | 78.111 | 81.278 |
| full-original | TTFT ms | 11563.233 | 11454.175 | 11564.713 |
| full-original | TPOT ms | 556.812 | 567.210 | 555.704 |
| full-original | 输出 tokens/s | 1.408 | 1.409 | 1.410 |
| full-original | MTP 接受率 % | 100.000 | 100.000 | 100.000 |

相对前后两轮baseline，decode128 TPOT降低 8.09%–10.32%，输出吞吐提高 7.82%–15.27%。此范围为两个对照点的结果，不是统计置信区间。

decode128 为四条固定短输入，每条生成128 tokens。full-original 为35269-token固定输入，自然EOS输出6 tokens；后者的 TPOT 不能代表稳态 decode。
长请求全部60份输出token IDs一致：True，序列为 [19186, 31515, 1522, 283, 2912, 248046]。
共享服务器、单个候选重启；A1/B1/A2控制了前后baseline漂移，但不等同于多轮ABBA、置信区间或SLA最大QPS验收。

## 精度与未关闭项

固定50题，temperature=0、seed=42、max_tokens=32，同一模板和请求。

| 服务 | C4 三次正确数/50 | C1 两次正确数/50 | C1 两次预测翻转数 |
| --- | --- | --- | ---: |
| baseline A1 | [40, 45, 41] | 未执行相同串行合同 | — |
| native B1 | [42, 40, 41] | [42, 39] | 5 |
| baseline A2 | [42, 42, 37] | [39, 39] | 2 |

C1保持max_tokens=32，增加top5 logprobs并逐条提交，答案token日志和逐题翻转另存。
baseline A1早期max_tokens=1探针只能观察首token，不用于答案精度比较。
baseline自身的重复变化说明不能把历史42→40/50全部归因于原生算子；这些小样本也不能证明模型精度等价，严格精度验收仍未关闭。

历史空槽位801-token输出有一次968/2465280元素差异，最大绝对误差0.00112915，原始失败保留。
本轮旧nullguard包五个新进程完整复测200项均通过，新modelmath包44项通过；这不构成该偶发问题已修复的证据。

## 分支、包与复现

- SDK：`${WORKSPACE}/xllm-ops-native-mtp-20260919`，分支 `perf/gdn-native-mtp-20260919`。
- 框架：`${WORKSPACE}/gdn-native-model-20260919/vllm-ascend`，分支 `experiment/gdn-native-sdk-model-20260919`。
- 新SDK：`${WORKSPACE}/xllm-ops-native-mtp-sdk-modelmath-20260919`；启动环境：`${WORKSPACE}/gdn-native-model-20260919/env-modelmath.sh`。
- 测量控制器：`diagnose_acceptance.py`；三轮实际命令保存在各自`command.sh`。
- SDK测试：`test_native_model_contract.py`、`test_mega_gdn_native.py`；框架NPU测试：`test_gdn_kv.py`。
- `summary.json` 保存完整指标、接受计数、输出 IDs、请求SHA；`serial-comparison.json`保存答案logprob及翻转；`source-and-binary-manifest.json`锁定源码及二进制。

代码保留为可审阅的分支工作树变更，未commit/push。三轮服务都通过进程、端口和NPU清理检查；生产安装、共享旧实验目录与卡0任务未改动。
候选运行的环境路径、服务路由日志均已归档；此调试控制器未采集候选worker的`/proc/maps`，不宣称完成正式加载身份验收。


---

以下为历史记录，保留当时的结果与限制。

## 后续定位与修复验证（2026-09-19）

MTP 接受率回退的主因已定位并修复：单序列 native prefill 在 slot 0 预热中
把写索引转换为 -1，却仍进入卷积状态写回，改写 conv bank 前方 15360 字节。
`ProcessSingle` 增加写索引边界保护后，正常原生状态池路径接受率恢复到
**78.3333%**（1081/1380），
短请求均值 TPOT 10.754 ms。CEval50 为 **40/50**，
baseline 为 42/50，有 2 题预测翻转，精度门禁 **FAIL**；
35269-token 原始输入的 4 个请求全部成功，输出 IDs 与 baseline 一致。

新 SDK 为 `${WORKSPACE}/xllm-ops-native-mtp-sdk-nullguard-20260919`，启动环境为 `${WORKSPACE}/gdn-native-model-20260919/env-nullguard.sh`。
SDK 完整复测 40/40；首轮 801-token 空槽位一次输出差异未定位，之后完整复测及
120 次定向对照通过，失败记录保留。这里确认接受率回退的修复，不宣称完成所有
数值稳定性与正式性能验收。诊断报告和复现证据在 `${WORKSPACE}/gdn-native-model-20260919/acceptance-diagnosis`。

以下为修复前候选报告，旧 FAIL 结果保留，不覆盖历史证据。

## 修复前模型接入记录（历史）

模型：Qwen3.6-27B W8A8_DYNAMIC；Ascend 910B3；TP4、MTP3、并发 4、FP32 SSM。
target 为 FULL_DECODE_ONLY，四个 draft wrapper 为 FULL；prefix cache 关闭，chunked prefill 开启。
框架分支 `experiment/gdn-native-sdk-model-20260919`，工作树 `${WORKSPACE}/gdn-native-model-20260919/vllm-ascend`。
SDK 分支 `perf/gdn-native-mtp-20260919`，加载 `${WORKSPACE}/xllm-ops-native-mtp-sdk-final-20260919`。
独立服务端口 8826，物理卡 4–7。测试结束后两个服务均已通过 cleanup 检查。

## 接入

decode / MTP 直接传入原生状态池、真实 token stride 和设备 metadata；prefill 直接读写原生 conv / SSM 状态。支持条件之外的混合 prefill/decode 仍保留框架 fallback。
四个候选 worker 的 maps 已验证加载目标 binding 和 SDK opapi，日志确认 native prefill 与 native decode 调用。
未修改共享实验目录或生产安装。代码未 commit/push。

## 正确性

框架单元测试 77 项、真实 NPU 测试 21 项通过；SDK 的 33 项原生测试与 54 项旧接口回归证据继续绑定相同源码、二进制哈希。另补测 conv/SSM 共用存储、非零 storage offset、1050 slots 和高编号 checkpoint，通过。
原生 S1 使用 MTP BF16 gate/readout 语义，因此旧 ordinary-decode 的两个接口 oracle 测试改为检查 MTP 第一 checkpoint；未放宽误差容限。

固定 CEval 50 题：baseline **42/50**，native **41/50**。
相对本次 baseline 新增错误 **1** 道，严格精度门禁 **FAIL**。
完整长请求四份输出 token IDs 一致：**True**。
这是 50 题 smoke，不是全量 CEval；历史 baseline 为 40/50，本次使用新鲜 baseline 的逐题结果，未复用历史分数。

新增错题及预测翻转保存在 `model-comparison-004.json`，原始请求和响应保存在各服务目录。之前候选 attempt 003 的同套源码也取得 41/50，新增错误为 advanced_mathematics 的 index 2、7。

## 性能

独立服务和图缓存；每场景预热 5 批，再测 5 批，每批并发 4。A1 来自 baseline attempt 002；初次候选 002 因日志证据检查中断并清理。修正日志及校验时机后，native attempt 003 完成长请求测量，但因 300 个 draft tokens 中接受数为 0 触发门禁。该失败保留为 FAIL_MTP_ACCEPTANCE，未改判 PASS。native attempt 004 补测短请求，复用 baseline 002。表中 TTFT/TPOT 为 20 个请求均值，吞吐统一按各测量批次时长总和计算，排除批次之间日志处理；启动、编译、capture 不计入。
长请求输入 35269 tokens、输出上限 9864，保留自然 EOS；decode128 为固定生成 128 tokens 的四条短输入。长请求通常仅输出 6 tokens，其 TPOT 不能代表稳态 decode。

| 场景 | 指标 | baseline | native | 变化 |
| --- | --- | ---: | ---: | ---: |
| full-original | TTFT ms | 11546.550 | 11428.614 | -1.02% |
| full-original | TPOT ms | 555.564 | 1049.556 | +88.92% |
| full-original | 输出 tokens/s | 1.411 | 1.402 | -0.65% |
| decode128 | TTFT ms | 546.979 | 476.510 | -12.88% |
| decode128 | TPOT ms | 10.838 | 27.141 | +150.42% |
| decode128 | 输出 tokens/s | 238.695 | 123.303 | -48.34% |

长请求 MTP 接受率：baseline 100.00%，native 0.00%。
短请求 MTP 接受率：baseline 77.98%，native 4.55%。
接受率大幅下降是本轮需要定位的模型级问题；不能用算子微基准收益替代模型结果。

本轮比较同时改变 GDN 状态布局与算子接入，测到的是整体接入效果。单轮 A/B、共享服务器，未做多次独立重启的 ABBA 或 SLA 最大 QPS 验收；上述数据为探索性结果。若严格精度门禁 FAIL，不能据此验收或发布性能优化。
MTP 接受计数、prefix-cache 零命中、每批时长范围及原始 SSE token 事件均已归档。

## 复现与证据

`scripts/service-command.sh` 为完整服务命令；`scripts/env-kv.sh` 固定源码和 SDK 路径。
`scripts/model-validation.py`、`scripts/perf-helpers.py` 保存本轮控制和评测方法。复跑使用新三位数编号，避免覆盖既有证据。
源码身份见 `framework-source-sha256.json`，SDK 哈希见前一阶段 `native-validation-manifest.json`。
本轮实际执行脚本与故障日志均保留；004 为补测，完整 A/B 复跑应使用新的编号。
完整汇总见 `model-summary-004.json`。


## 修复前结论与待定位点（历史）

接入和测试已完成，模型验收 FAIL。两次候选同为 41/50，但新增错题不同：003 为高等数学 index 2、7，004 为操作系统 index 9（D→A）。这说明一次总分不能代替逐题及重复运行验证；尚不能把每个预测翻转都归因于同一算术问题。
短请求接受率从 77.98% 降到 4.55%，TPOT 从 10.84ms 增至 27.14ms。长请求输出 IDs 相同，但接受率从 100% 降到 0%。这些模型结果没有兑现前一阶段算子微基准的收益。
后续应优先对照 draft 与 target 的实际输入、状态继承、accepted checkpoint 和输出 logits，定位接受率回退；本轮尚未确认根因，不能宣称已修复。
未修改算子二进制或生产安装；没有 commit/push，容器保持运行。

## 发布分支

框架发布至 `Sinle4Cat/vllm-ascend:adn_gdn_new`，算子发布至 `Sinle4Cat/xllm-ops:adn_gdn`。上文保留历史验证记录；`${WORKSPACE}` 表示验证时使用的工作目录，历史实验 SDK 不是默认发布包。最终算子 H/O 上限及验证结论见 xllm-ops 的 native MTP 验证文档。
