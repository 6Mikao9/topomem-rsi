# TopoMem-RSI SGLang v3 正式结果

在冻结的 SGLang 0.5.20 attention-DP2 双 rank 目标执行两轮搜索、冻结和 12 块留出测试。每块测量两案并汇总两案窗口。两次生成器修订均完成行为对照并被拒绝。第一轮未达到稳定配对改进门槛，第二轮的子臂保留 incumbent。最终 active generator 与 adaptive、fixed 两个入选程序的真实来源均为 seed。

| 留出对照 | adaptive 窗口缩短块数 | 配对中位差 ms | 达到预登记门槛 |
|---|---:|---:|:---:|
| 原生默认 4096 全 host | 7/12 | +10.957513 | 否 |
| 固定生成器方案 | 0/12 | −32.568516 | 否 |
| 训练选优手写方案 | 0/12 | −26.065438 | 否 |

正值表示 adaptive 更快。门槛要求每块正增益且中位差严格大于 11.49229000000014 ms。冻结 adaptive 在两案均以 4096 slots、每 rank 第 0 层本地 HBM 加五层 pinned host 运行。固定生成器和手写方案在两案均运行 resident 6144。补充试验重用两案，将 adaptive 与原生调优容量 6144 全 host 配对，5/8 块窗口缩短、中位差 +11.923439 ms，未通过稳定增益门槛。[正式原报告](controller/report.json) · [调优容量补充对照](../sglang-native6144-ablation-v1/README_ZH.md)

实际完成 20 个模型调用，提供方报告 282,471 tokens，自适应和固定臂分别报告 136,466 与 146,005 tokens。226 个租约作业含训练 34、验证 82、测试 110。服务器逐原件复算所有 226 个全局转发及 ingress 记录，另复算 40 次资格阶段双 rank 的 packed KV 摘要。运行使用架构测试用 toy GLM 模型与固定请求。[审计](export_audit.json) · [服务器证据](server-evidence/README_ZH.md)

## 工件

- [冻结选择及来源](selected/selection.json) · [adaptive 程序](selected/adaptive.program.json) · [fixed 程序](selected/fixed.program.json) · [实际生成器](selected/adaptive.generator.py)
- [原控制器 journal](controller/journal.sqlite) · [原冻结记录](controller/frozen.json) · [原 API 和计划](original-api-and-plans.zip) · [原租约回执 ZIP](leased-receipts.zip)
- [四个固定 case](formal-cases/README_ZH.md) · [公开 adapter contract](adapter_contract.json) · [参数化 schema 3](schema3.template.json) · [冻结源码 ZIP](frozen-source.zip)
- [本地复核和查看器](REPRODUCE_ZH.md) · [工件摘要清单](artifact_manifest.json)
