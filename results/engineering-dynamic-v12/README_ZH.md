# CUDA GPU3 动态特征 lowering 工程对照

八个交错区块比较原生 LRU、v11 writeback 和同一 writeback IR 的 v12。四个已有公开训练场景共享 GPU3、CPU64、三次预热和五次计时。该结果估计执行器差异，属于固定公开输入的工程对照。

| 场景 | v12 较 v11 更快的块 | v11 − v12 配对中位差 ms | v12 较原生更快的块 | 原生 − v12 配对中位差 ms | 旧差距收回比例 |
|---|---:|---:|---:|---:|---:|
| 四案汇总 | 0/8 | -9.900174 | 0/8 | -59.940892 | -19.7% |
| agent-water-inspection | 2/8 | -0.313324 | 8/8 | +1.681853 | — |
| chatbot-trip-planning | 1/8 | -0.143448 | 7/8 | +0.709746 | — |
| long-document-estuary-qa | 0/8 | -9.396743 | 0/8 | -61.198989 | -17.9% |
| rollout-bridge-options | 3/8 | -0.063627 | 0/8 | -1.036220 | -6.1% |

正值表示 v12 窗口缩短。旧差距收回比例是事后描述量，分子为配对中位差 v11 − v12，分母为正的配对中位差 v11 − 原生。旧方案没有正差距的场景用横线表示。预注册没有规定一半为验收门槛。

两版 writeback 的有序动作、生命周期、选中 KV、注意力输出、缺页、链路字节和容量逐案相同。原生对照匹配固定输入和选中输出。

[预注册](inputs/preregistration.json) · [配对统计](paired-summary.json) · [原始分析](run/analysis.json) · [回执 ZIP](leased-receipts.zip) · [工件清单](artifact_manifest.json)

本地复核可以从公开目录执行。

```sh
python -B tools/export_v12_gpu3_engineering.py verify --publication . --replay-analysis
```
