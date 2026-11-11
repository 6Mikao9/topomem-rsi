# SGLang RSI v2 训练阶段中止

原计划在冻结的 SGLang resident v2 目标上执行两轮搜索，随后冻结候选，最后运行封存测试。控制器于 **2026-11-10 19:58:07 UTC** 启动，因实际提示存在互相冲突的规则，于 **20:13:45 UTC** 终止。没有完成搜索轮，未进行候选配对验证，未冻结或执行留出测试。[原计划](pipeline-plan.json) · [原中止记录](abort-audit.json) · [退出记录](pipeline-status.json)

## 记录中的冲突

| 证据 | 内容 |
|---|---|
| 实际提供给模型的结构化规则 | `trusted_program_contract.required_rule` 无条件要求 `allocate/place` |
| 同一次请求的可信目标说明 | resident 模式要求省略全部 `allocate/place` 规则 |
| `r0000/parent/00` 的实际程序 | 包含 `configure/hot_capacity`、`configure/backing_mode` 和 `allocate/place` |
| 编译结果 | `resident backing cannot include cold placement rules`，未发出该候选的 GPU 作业 |

使用同一冻结编译器和唯一训练 case 离线重现了完全相同的拒绝。实际输出与提示冲突一致，记录无法确定模型添加该规则的动机。第二个父生成器候选 `r0000/parent/01` 编译为 4096 slots、tiered，训练窗口 1082.140216 ms，同一训练 case 的原生窗口为 1075.581884 ms。单次训练测量没有形成比较结论。[编译与成本审计](audit.json) · [原 API 与计划记录](original-api-and-plans.zip)

## 实际执行边界

- 完成 4 个 API 调用，提供方合计报告 **62,565 tokens**，已完成调用耗时合计 **449.750 秒**。自适应臂报告 48,461 tokens，固定臂报告 14,104 tokens。首个 child task 请求在本地 journal 保留 `pending`，提供方是否完成及其成本未知。归档保留已返回的最终内容和 usage。
- 10 个实际 GPU 作业的本地和服务器证据闭合，训练 7 个，预先规定的原生验证基线 3 个。后者含 A/A 资格及一次计时，未进入实际模型请求上下文。冻结可信训练上下文摘要在 5 条 API journal 中逐一匹配，实际请求未包含验证或留出的 case ID 或 case 专属摘要。
- 10 次作业从双 rank 原始事件复算全局调度和 ingress，6 次资格从选行日志复算 packed KV 摘要。相应原始服务器文件均在 `server_raw/`，含原请求、进程日志、编译与行校验记录。这里的 packed KV 摘要来自 GPU 诊断记录，原始显存快照不属于该记录格式。[服务器证据与逐成员摘要](server_raw/index.json)

## 归档与复核

- [SQLite 原 journal](controller/journal.sqlite) · [原 controller manifest](controller/manifest.json) · [原 API 请求和 meta 计划记录](original-api-and-plans.zip) · [原 leased receipts](leased-receipts.zip)
- [冻结本地源 ZIP](frozen-source.zip) · [冻结源清单](frozen-source-manifest.json) · [公开 adapter contract](adapter_contract.json) · [参数化 schema 3 模板](schema3.template.json)
- [工件 SHA-256](artifact_manifest.json) · [训练手写对照和部署约束](../sglang-hand-sweep-v2/REPRODUCE_ZH.md)

部署方私有 schema 3 原件的 SHA-256 为 `a4c62726543fa19ac815ddc8e5f21388789231940306df29c994f4e592813740`。公开模板替换执行主机、路径和 UUID，供新部署重绑。原 journal 连同当时的 lease 状态只作取证。原计划中的 `freeze` 与 `test` 阶段没有执行，归档因此没有选中策略或对应生成器产物。v2 运行已终止，后续修订应使用独立 run、配置和源指纹。
