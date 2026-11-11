# CUDA GPU3 零跳 producer 保留工程对照

八个交错区块比较原生 LRU、v11 LRU 等价 IR 及同一 IR 的 v13。四个已有公开训练场景共享 GPU3、CPU64、三次预热和五次计时。该结果估计执行器差异，属于固定公开输入的工程对照。

| 场景 | v13 较 v11 更快的块 | v11 − v13 配对中位差 ms | v13 较原生更快的块 | 原生 − v13 配对中位差 ms | 旧差距收回比例 |
|---|---:|---:|---:|---:|---:|
| 四案汇总 | 8/8 | +11.661276 | 0/8 | -11.449308 | +49.3% |
| agent-water-inspection | 8/8 | +1.739127 | 0/8 | -3.512217 | +34.8% |
| chatbot-trip-planning | 8/8 | +2.116189 | 0/8 | -3.917015 | +35.9% |
| long-document-estuary-qa | 8/8 | +6.599623 | 0/8 | -2.326846 | +71.1% |
| rollout-bridge-options | 8/8 | +0.873592 | 0/8 | -1.986983 | +30.4% |

正值表示 v13 窗口缩短。旧差距收回比例是事后描述量，分子为配对中位差 v11 − v13，分母为正的配对中位差 v11 − 原生。旧方案没有正差距的场景用横线表示。预注册没有规定一半为验收门槛。

两版 LRU 等价 IR 的有序动作、生命周期、选中 KV、注意力输出、缺页、链路字节和容量逐案相同。原生对照匹配固定输入和选中输出。八块逐案确认合计 1,794 次通用单候选保留变为 1,794 次静态保留，其他诊断工作计数相同。

[预注册](inputs/preregistration.json) · [配对统计](paired-summary.json) · [原始分析](run/analysis.json) · [回执 ZIP](leased-receipts.zip) · [工件清单](artifact_manifest.json)

本地复核可以从公开目录执行。

```sh
python -B tools/export_v13_gpu3_engineering.py verify --publication . --replay-analysis
```
