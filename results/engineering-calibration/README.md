# v5 与 v6 CUDA 工程校准

离线核对了 13 份公开资格回执。v5 有四个单次独立进程，v6 有三对原生 A/A 独立进程，以及原生等价 IR、直接手写和手写 IR 各一次。每个回执对每个用例有三次计时。A/A 按 A1、B1、A2、B2、A3、B3 顺序完成。以下为描述统计，三对 A/A 没有用来估计置信区间或假阳性率。

| 用例 | 原生 A/A 绝对差中位数 / 最大值（ms） | v6 原生（ms） | v6 LRU 等价 IR（ms） | v6 直接手写（ms） | v6 手写 IR（ms） | 本次最快的独立对照 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 工具代理 | 0.190 / 0.239 | 48.391 | 56.366 | 50.054 | 62.962 | 原生 |
| 对话增长 | 0.564 / 0.668 | 58.121 | 68.154 | 68.146 | 75.289 | 原生 |
| 长文问答 | 1.142 / 1.617 | 174.785 | 210.901 | 181.035 | 263.693 | 原生 |
| 分支 rollout | 0.230 / 0.439 | 21.470 | 25.171 | 20.279 | 28.526 | 直接手写 |
| 四用例汇总 | 1.731 / 1.931 | 302.850 | 360.591 | 319.515 | 430.471 | 原生 |

表中 v6 原生为六个独立回执的中位数。每个回执的汇总分数为各用例计时中位数之和。每个用例的完整样本和 v5 绝对时间保存在 JSON。最快对照只是本次校准观测，直接手写的单次结果尤其需要在保留集和候选之间重新配对。

原生等价 IR 的汇总从 v5 的 1,409.639 ms 降到 v6 的 360.591 ms，与对应原生的比值从 3.193 降到 1.191。v6 通过驻留及退休索引和编译后的规则、路径筛选，减少了固定执行器的开销。两版的各臂程序、执行签名、注意力选择和输出、链路字节及缺页计数一致。此为跨版本、非交错的工程比较，源清单分开保存。计时差不能归因于某个单一代码改动，也不能计入模型自我改进。

资格范围为两张匹配的 RTX 5090 上的文本派生 CUDA 块稀疏注意力。回执绑定有序设备身份、源清单、用例和结果哈希，并记录完成和进程退出。公开摘要只保留 GPU UUID 指纹。进程起止检查的证据不覆盖整个运行窗口中的临时外部负载或固定频率。受管预留、受管占用峰值与 PyTorch 张量分配峰值分别保留。

原始回执已由控制端保管，可用相同回执重建摘要。

```powershell
python -B results/engineering-calibration/build_calibration.py --v5 <public-v5-dir> --v6 <public-v6-dir> --output <new-json>
```

## 留出分析的预先登记

离线工具 `rsi_memory.statistics` 只读取已结束的配对摘要。检查留出结果前，应封存分析计划与 `rsi_memory.topology.fingerprint(plan)` 的值。计划包含 `schema=1`、`kind="paired_latency_preregistration"`、12 个 `block_ids`、全部 `case_names`、`candidate`、全部 `comparators`、逐用例和 `aggregate` 的 `threshold_ms`，以及固定的 `bootstrap` 参数 `seed`、`draws`、`family_alpha`。实用收益阈值由实验登记确定。上述 A/A 差值可辅助选定阈值，证据范围仅为本次公开用例。

以原生或手写对照时间减候选时间作为每个匹配块的收益。工具报告所有用例和对照，包括失败或缺失臂。完整的 12 块使用同一重采样索引计算配对中位数的百分位区间，并对预先指定的全部对照、用例和汇总值作 Bonferroni 调整。区间基于匹配块可交换的近似，和控制器自己的准入规则分别报告。未在同批配对的对照不可拼接到同一试验。

```powershell
python -B -m rsi_memory.statistics --plan <frozen-plan.json> --expected-plan-sha256 <predeclared-sha> --paired-result <settled-pair.json> --output <new-analysis.json>
```

## 发布证据与复算

| 归档标签 | 内容 | 证据范围 |
| --- | --- | --- |
| `v5-v6` | v5 四份、v6 九份原始回执 | 人工 CUDA 执行器工程校准，含原生 A/A |
| `v10-paired` | 32 份原始回执及 [公开预登记](topomem-v10-paired-preregistration.json) | GPU3 单卡，八个配对块，四个公开训练用例 |
| `v11-paired` | 32 份原始回执及 [公开预登记](topomem-v11-paired-preregistration.json) | 同一单卡工程设计的独立复验 |
| `v8-public` | 12 份回执、一份链路 profile、一份嵌套源码 ZIP | 原始 v8 工程归档的公开派生版本 |

每份成员保留原始字节，归档 SHA-256、成员 SHA-256 及字节数见 [发布清单](receipts-manifest.json)。v8 原 ZIP SHA-256 为 `a67055baa7c2a1f692920ab94045333a3acda918ec4e67b2697fd8c25997d7fa`。派生版仅排除带操作者网络地址的 `qualify_geometry_v8.py`，排除成员 SHA-256 为 `35ab56d46430b85b6b12f26bce485fdf0753d4820a09d37ef6c434d0064c9a17`。原包和 [原始摘要](cuda-v8-single-gpu-geometry.json) 在本地保留。其他归档省略工作目录的连接日志。

从仓库根目录运行。`OUT` 指向自己新建的工作区，`$OUT/engineering` 和三个摘要输出必须是新路径。可省略提取命令的 `--output` 只校验，不写文件。三个摘要均已用归档中的原始回执复算至字节一致。

```sh
E=results/engineering-calibration
OUT=/path/to/new-work
python -B "$E/extract_receipts.py" --manifest "$E/receipts-manifest.json" --output "$OUT/engineering"
python -B "$E/build_calibration.py" --v5 "$OUT/engineering/v5-v6/v5" \
  --v6 "$OUT/engineering/v5-v6/v6" --output "$OUT/v5-v6.json"
python -B "$E/build_v10_gpu3_calibration.py" --receipts "$OUT/engineering/v10-paired" --output "$OUT/v10.json"
python -B "$E/build_v11_gpu3_calibration.py" --receipts "$OUT/engineering/v11-paired" \
  --preregistration "$E/topomem-v11-paired-preregistration.json" \
  --v10-audit "$E/engineering-v10-gpu3-qualification.json" --output "$OUT/v11.json"
```

v10 原启动脚本含私网地址，发布版本改为由操作者传入私有执行配置。参数化版本使用已由 v10 审计确认的 v11 公共用例载荷，校验 v10 源清单、原脚本摘要、用例、页几何、预热次数及配对顺序。以下只验证重建。加 `--execute --controller-root <local-source> --executor-config <private-json> --audit-directory <new-dir>` 才提交独立 GPU 作业。v11 的原参数化脚本和原计划保留，可按工程文档用新结果目录执行。回执复算只读已有证据，新的 GPU 计时需另行记录设备及运行环境。

```sh
python -m zipfile -e "$E/topomem-src-v10-qualification.zip" "$OUT/v10-source"
python -B "$E/replay_v10_gpu3.py" --v10-preregistration "$E/topomem-v10-paired-preregistration.json" \
  --v11-preregistration "$E/topomem-v11-paired-preregistration.json" \
  --source-manifest "$OUT/v10-source/SOURCE_MANIFEST.json"
```

v5 和 v10 源包在本目录。v6 和 v11 单卡资格源码见 [源码快照清单](../source-snapshots/manifest.json)。这些样本来自人工工程和固定训练对照。它们用于核验执行器和比较开销，没有模型生成器晋级或留出收益的证据。v10 和 v11 的运行窗口不同。 [v7 至 v11 工程审计](ENGINEERING_V7_V11_ZH.md) · [v10 摘要](engineering-v10-gpu3-qualification.json) · [v11 摘要](engineering-v11-gpu3-qualification.json)
