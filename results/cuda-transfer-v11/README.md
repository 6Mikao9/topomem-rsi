# CUDA v11 四场景生成器迁移

本实验冻结 v7 的初始生成器与 R2 晋级生成器，在 v11 双卡 CUDA 文本派生块稀疏注意力执行器上各调用八次 `build_prompt`。四场景各有两个生成槽和两份独立留出用例。Agent 复测 v7 已用过的留出内容，其他三个场景沿用各自预先封存的用例。两源均使用固定的 `round=2` 入口，本实验没有新增 RSI 轮次。 [预登记](preregistration.json) · [独立审计](audit/AUDIT_ZH.md)

220 份原始 GPU 回执包括 12 份训练锚点、16 份训练候选和 192 份留出作业。两项训练候选因 `InfeasiblePlan` 被拒绝，均保留。每个留出区块包含两源的封存程序、项目 CUDA 直写/LRU 和直接手写四臂。48 个区块的选中 KV 与注意力输出核验均通过。晋级源的产出在四场景的 48 个区块均慢于 CUDA LRU，四场景相对初始源均未通过严格门槛。两源封存程序的 96 个配对用例中，有序物理动作、生命周期、链路字节、缺页、受管预留和峰值均相同。四场景留出均未在第二张 GPU 存放或传输 KV。 [封存选择](selected.json) · [留出数据](heldout.json)

Chatbot 的晋级源产出在 12 个区块均较初始源快，配对增益中位数为 6.295 ms，最小为 0.819 ms，低于每块须超过的 1.5 ms 门槛。两份 Chatbot 用例中，初始源的诊断分别执行 356 和 296 次通用放置选择，扫描 1,068 和 888 个候选，晋级源分别执行 356 和 296 次静态路线选择。物理轨迹相同，工作计数支持静态 lowering 路径不同。计时未单独分离解释器成本。 [工作计数与逐区块审计](audit/audit.json)

## 重建证据与离线浏览

原始台账、空字节 lease 文件、生成及选择封存、试验 driver 和独立审计逐字节保留。连接日志留在原工作目录。原始回执以 ZIP 保存，可核对归档和每成员 SHA-256。`viewer/` 内的四组 topology 和 workload 来自各场景实际训练用例的首条 `cases[0]`，属于展示派生文件。原始案例保持完整，派生索引记录原训练文件和 case SHA-256。 [发布清单](evidence-manifest.json) · [展示来源](viewer/manifest.json) · [冻结运行时源码](../source-snapshots/manifest.json)

从仓库根目录运行，指定新的 `OUT/transfer` 提取目录和 HTML 输出路径。命令只校验、提取和渲染已有证据。

```sh
T=results/cuda-transfer-v11
E=results/engineering-calibration
OUT=/path/to/new-work
python -B "$E/extract_receipts.py" --manifest "$T/evidence-manifest.json" --output "$OUT/transfer"
python -B -m rsi_memory.topology_report --transfer "$T" \
  --receipts "$OUT/transfer/transfer-v11" \
  --topology "$T/viewer/agent_tools-topology.json" --scenario "$T/viewer/agent_tools-workload.json" \
  --topology "$T/viewer/chatbot_growth-topology.json" --scenario "$T/viewer/chatbot_growth-workload.json" \
  --topology "$T/viewer/long_document_qa-topology.json" --scenario "$T/viewer/long_document_qa-workload.json" \
  --topology "$T/viewer/rollout_cow-topology.json" --scenario "$T/viewer/rollout_cow-workload.json" \
  --max-actions 32 --output "$OUT/cuda-transfer-v11.html"
```

需要重新配置运行时，可解包 `../source-snapshots/topomem-v11-integration.source.zip`，核对嵌入的 `SOURCE_MANIFEST.json` 和 driver 摘要，再按 [原始预登记操作说明](README_ZH.md) 提供本机路径和私有执行配置。历史台账用于查阅。新试验使用独立目录和预算。这里的质量摘要验证固定文本派生注意力和内存生命周期，真实模型服务指标需独立证据。
