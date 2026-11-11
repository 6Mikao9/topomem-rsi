# TopoMem CUDA v4.1 工程复评

四个已公开文本场景均在同一张物理 GPU3 上执行 H64 注意力，逻辑设备名为 gpu0。KV 预算为 128 KiB GPU 和 32 MiB pinned host。该工程复评与此前中断的双 GPU 试验分别归档。

两轮各使用两个任务槽。控制器先按训练测量封存程序，再在六个配对块上评估。三个封存角色分别记录全局训练管理器、已晋级来源的输出及固定生成器输出。

| 封存角色 | 试验 | 生成器 SHA-256 | 重复训练中位数 ms |
|---|---|---|---:|
| manager_best | r0001/child/00 | 7ef5507ac6345f80f0ea82b3a747fb39d67d767cbecae3c42ed1eb51012dcd62 | 296.643452 |
| admitted_output | r0001/parent/01 | 42699e4d5148ee3dc3f41f7ffeab8a3020e8fb75a5a433940d7ae88e4c08ca38 | 301.021342 |
| fixed | r0000/fixed-a/00 | 42699e4d5148ee3dc3f41f7ffeab8a3020e8fb75a5a433940d7ae88e4c08ca38 | 286.020065 |

| 正式对照 | 缩短窗口的区块 | 配对中位差 ms | 通过门槛 |
|---|---:|---:|:---:|
| manager_best_vs_native | 0/6 | -37.738890 | 否 |
| manager_best_vs_fixed | 0/6 | -15.535971 | 否 |
| manager_best_vs_manual | 6/6 | +35.415492 | 是 |
| admitted_output_vs_native | 0/6 | -36.515972 | 否 |
| admitted_output_vs_fixed | 0/6 | -15.476750 | 否 |
| admitted_output_vs_manual | 6/6 | +36.099599 | 是 |

两轮中晋级 0 次。完整审计列出每轮计划、决定、实际源复用及所有训练重复和每场景内存指标。核对了 134 份租约回执和 20 个实际模型请求。

v4.1 提示修订与 v13 零跳保留优化由操作方提供。原生、人工管理器、初始程序及生成程序使用同一 v13 执行器。此前 24 份工程资格回执属于独立对照。实际发给模型的 v4.1 提示说明父、子各自从新槽选出候选，历史 incumbent 独立参与配对验证。

[原终局报告](controller/report.json) · [独立审计](audit.json) · [原始回执 ZIP](leased-receipts.zip) · [公开配置模板](inputs/adapter-config.template.json) · [工程资格绑定](qualification/binding.json) · [工件清单](artifact_manifest.json)

离线校验命令。

```sh
python -B tools/export_topomem_v41_gpu3.py verify --publication .
```
