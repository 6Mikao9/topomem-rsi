# SGLang resident v2 手写策略选优审计

在唯一的训练 case `archive-train-reproduction` 上，用冻结的 SGLang 0.5.20 attention-DP2 双 rank 目标比较九个预登记手写策略。每次观测启动独立进程，复用固定请求、token 承诺、ingress、调度和显存预算。此选优为后续正式实验冻结手写对照，数据范围仅训练 case。模型 `inference-optimization/GLM-5.3-0.6B-A0.4B` 用于架构验证，实测涵盖真实 attention 与 KV 消费，未评价任务质量或生产环境吞吐。

## 配对结果

表内为 `stock-6144-host` 的测量窗口减去该策略窗口，单位 ms，正值表示该策略更快。计时以预热后整窗为单位，每块只有一个训练 case。候选需三块均为正且中位差至少 5 ms，再按中位差选优。三个块使用预登记 Williams 顺序的前三行，完整循环共 18 行。

| 策略 | 第 1 块 | 第 2 块 | 第 3 块 | 中位差 | 达到选优门槛 |
|---|---:|---:|---:|---:|:---:|
| stock-4096-host | +19.584577 | -3.576525 | -4.300594 | -3.576525 | — |
| stock-6144-host | +0.000000 | +0.000000 | +0.000000 | +0.000000 | — |
| local-layer-0 | +24.884628 | -0.335319 | -5.849864 | -0.335319 | — |
| local-layer-1 | +17.339690 | -0.937016 | -7.891476 | -0.937016 | — |
| local-layer-2 | +20.171098 | -3.380522 | -0.092550 | -0.092550 | — |
| local-layer-3 | +12.452786 | +6.878734 | +0.341528 | +6.878734 | 是 |
| local-layer-4 | +10.829301 | +3.521631 | -4.054233 | +3.521631 | — |
| local-layer-5 | +14.348008 | -3.547097 | +3.917530 | +3.917530 | — |
| resident-6144 | +41.331836 | +21.610484 | +22.233877 | +22.233877 | 是 |

`resident-6144` 胜出，三次窗口分别为 **1057.218131、1059.524526、1057.597642 ms**。相对原生默认 `stock-4096-host` 的逐块差为 **+21.747259、+25.187009、+26.534471 ms**。原生默认训练窗口的极差为 **5.74614500000007 ms**，由预定公式 `max(5 ms, 2 × range)` 得到正式实验的晋级门槛 **11.49229000000014 ms**。手写选优使用的门槛为 5 ms。

选中程序的规范 JSON SHA-256 为 `6a02e87675b3bb78cba3ddb2934fa0ab9f6583e827cd1be96bc0ca7e70ecbbad`。[原始选优结果](sweep/result.json) · [原始选中程序](sweep/selected_program.json) · [逐块窗口和重算值](audit.json)

## 资格与资源核对

- 38 次作业全部完成，4 次为 4096/6144 原生同容量 A/A，7 次为非原生手写策略资格，27 次为计时。根据回执重建原始请求摘要和作业序列，执行时间无重叠，未报告超时、环境冲突、残留进程组或 GPU 计算进程。冻结代码离线重放逐行匹配原始账本及最终选择。
- 38 次作业报告相同的实际全局转发摘要 `54032fc74920338c8a6a617e5eee539160b1e8783626af1a4c6a48ae439e83a0`，各有 132 个 active forward，rank-local IDLE 计数为 0。调度、dispatch、输入 token、输出 ID、运行环境摘要逐项一致。11 次资格作业的两个 rank 各自匹配 packed KV 摘要，包括跨容量及 resident。27 次计时作业不生成 packed KV 校验、选行 miss 字节校验或 producer oracle 记录。
- 每 rank 的公共 GPU packed KV 池均为 **64,739,328 B**，由 6 层 × 16,448 物理行 × 656 B 得到，逻辑容量为 16,384 行。6144 的实际热元数据为 **879,616 B/rank**，比 4096 多 **290,816 B/rank**。各策略使用相同的增量容量上限 **25,715,808 B/GPU、266,612,736 B/host**。公共池与编译预算分列。
- 原生两档的冷池 pinned payload 为 **129,226,752 B/rank**。六个逐层手写方案各把同编号的一整层放在两个 rank 各自的本地 HBM，实测 **21,537,792 B/rank** 本地冷池、**107,688,960 B/rank** pinned 冷池，六个层位均有非零写回。对照不使用 peer HBM 或 relay。resident 复用公共池，冷 pinned payload、冷 HBM payload 及冷写回均为 0。sentinel 为逻辑 **3,936 B/rank**、Torch allocated 增量 **6,144 B/rank**、reserved 增量 0，编译保守计费 **12,582,912 B/rank**，连同增量热元数据合计 **12,873,728 B/rank**。
- resident 资格逐 rank 报告 756 次 swap 和 756 个新行的独立 producer 地址及字节校验，每个请求有 63 次 decode 校验。每 rank 的 prefill 校验行数按原始 token 长度和六层重新核对。原始服务器逐事件 KV/global/worker 日志已单独归档。本目录从 38 个原作业重算全局 tick 和 ingress，从 11 个资格作业重算 packed KV 摘要，逐项核对本地 leased receipt、原请求和结果 SHA-256。原始显存快照不属于该记录格式。[服务器证据与逐成员摘要](server_raw/index.json)

## 原件

- [原始预登记清单](sweep/manifest.json) · [原始执行账本](sweep/jobs.jsonl) · [38 份未改字节的 leased receipts](leased-receipts.zip) · [工件与 ZIP 成员 SHA-256](artifact_manifest.json)
- [完整公开 adapter contract](adapter_contract.json) · [去部署位置的 schema 2 模板](schema2.template.json) · [离线核对和参数化重放](REPRODUCE_ZH.md)
- [同一 worker source 的原生校准资格与 profile](../sglang-qualification/ingress-resident-v2/AUDIT_ZH.md)
