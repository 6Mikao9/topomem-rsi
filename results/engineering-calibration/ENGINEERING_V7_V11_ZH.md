# v7 至 v11 执行器工程审计

这是隔离集成草案。正式试验仍要求至少两张不同的 GPU，头维度上限为 256。受管 KV 预算、注意力、外部事件、计时边界与验收规则延续原契约。诊断运行启用工作量计数，训练反馈只投影预先允许的事件和数值字段，计时运行独立执行并拒绝意外计数。完整本地测试共 74 项通过。

## 源码与证据

| 版本 | 执行器改动 | 已核对证据 |
| --- | --- | --- |
| v7 | `demote` 在实际容量压力下迁移活跃权威块，支持向有效副本移交所有权 | 双卡 GPU0/1 五份资格回执完成，包括 pinned host、经 host 中转与已有副本交接。工程资格源清单 `fcf06725c1582a27afd526b5cc24ce77c26fe9eadbf472536528b523c9867574`。 |
| v8 | 按阻塞源一次检查路由，先处理能胜出的规则组，只读取规则实际使用的特征 | 84 组公开 CPU 路径与 v7 等价。另有 GPU3 单卡、三种头宽的 12 份工程回执。单卡资格源清单 `36f6af3b96581744199e55b4ba8121985985843ddabfa27dae04bdc99af81bce`。 |
| v9 | 按节点索引活跃副本，用静态条件过滤候选节点，新增可选的诊断工作计数 | 84 组 CPU 证据与 v8 逐字节相同。配对 CPU 工程计时 `v9-public-timing.json` SHA `72c3212a256a473f4a5685efd15e6abd49fae567c61592a78d620248c742f7d4`。 |
| v10 | 静态条件和可证明有限的状态分数使用稳定的索引最小值，其他规则使用原解释器 | 84 组公开 CPU 证据一致。配对 CPU 计时 SHA `f356f9b9056aabbd9f59471709fd3da47be4c40bd4bfeadbc688bd577636291f`。下表是另一批真实单卡 GPU 工程记录。 |
| v11 | 预计算全静态路由规则的优先级和延迟抛出的算术错误 | 84 组 CPU 证据一致。配对 CPU 计时 SHA `ca2d0bd513c51fd12a754776733b09354d520f7b0361859c37fa3b57e1a38a3f`。单卡复验已完成并在下文记录。 |

v7 到 v11 独立草案的运行时源码 SHA-256 依次为 `3cdf9a4eef034f33717e0e065ad9460a1002beeaba2c8df45bd32fa3b84340aa`、`5d40db4bb2863349d52f89e61c7eca5eca7474d9419c760d8a4a3d1639783923`、`c6cfc7cf17a567e7f178096d58c81ab804529e2af2662767ee8ba12e8be296df`、`20c65b644ddb00a60202b53a1426f70bf148811610cac0b5d9b21a0b8f6f2827`、`3bb7a74a8ac96f31d7497c576e3b654edb4ba2d7f45b2be8db11b9ed579d8ecd`。不同资格包的源清单单独记录。公开 CPU 轨迹从 v7 到 v11 共用摘要 SHA `ca6a856400eaff561b5fc9c13243a8cf7461bdeb9f2a1f1064328c975cea5402`。

本集成源包共 63 个成员，归档 SHA `1ed35df6bb3a05407205a5f10d40adefa4cb5f0499c30dc82ef80c3ef4c1212f`，源清单 SHA `877873960cd4da3c332891495fdc86d30fa59725b879d0fcffccb81fdef0c767`。三份运行时辅助源码和控制器投影均在清单中逐文件绑定。

## v10 的 GPU3 配对工程校准

回执目录为 `D:/ws/.scratch/topomem-qualification-v10-paired-gpu3`，源清单 SHA `68fe058da35c3b0517b6a2a769077693cea0774277ebdb8cd420958f4c774367`。审核逐一校验 32 份完成回执和不同进程身份。8 个 Williams 配对块的位置和一阶顺序均衡。四臂均使用固定四份公开训练用例，每个用例有五个计时样本。审核从诊断访问记录重算所选 KV 与输出指纹，从动作和生命周期重算执行签名，再与计时记录比对。原生与 LRU 等价臂的链路字节和缺页数逐用例逐块一致。

| 用例 | 原生中位数（ms） | LRU 等价（ms） | 直接手写（ms） | 写回（ms） | 原生减写回配对中位数（ms）及胜出块 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 代理 | 47.441 | 52.927 | 47.140 | 46.973 | +0.490，7/8 |
| 对话 | 57.047 | 63.893 | 63.720 | 57.778 | −0.944，1/8 |
| 长文 | 150.408 | 161.054 | 149.727 | 205.307 | −54.728，0/8 |
| Rollout | 21.106 | 24.229 | 19.194 | 22.643 | −1.581，0/8 |
| 汇总 | 276.190 | 302.326 | 279.917 | 332.698 | −57.150，0/8 |

直接手写在 rollout 的原生减对照配对中位差为 +1.883 ms，8/8 块同方向。汇总中位数来自每块四个用例时间之和。完整比较和原始回执哈希见 [工程校准 JSON](engineering-v10-gpu3-qualification.json)。这是 GPU3 单卡与 pinned host 的公开训练工程校准，证据范围只涵盖这批固定输入。进程起止检查无法覆盖运行窗口中的瞬时外部负载或实际固定频率。这里没有留出评估，也没有模型自产策略晋级。

```powershell
python -B docs/build_v10_gpu3_calibration.py --receipts <original-public-receipt-directory> --output <new-audit-json>
```

## 已冻结的 v11 单卡复验计划

单卡资格包只把最少 GPU 数从 2 调整到 1。逐成员比对确认运行时源文件只有这一行变化，头维度上限仍为 256。计划绑定 GPU3、CPU64、头维度 64、每臂三次预热及五个样本，并保留 v10 的 8 个 Williams 配对块。四份用例与两份 IR 的指纹来自实际 v10 回执。四个臂为原生 LRU、LRU 等价程序、直接手写和压力写回。

| 冻结产物 | SHA-256 |
| --- | --- |
| `topomem-v11-single-gpu3-qualification.source.zip` | `afca095a53ddd69f038ff7a9806e76da72f8b7411a54b8448ad3494dfedb6aac` |
| `topomem-v11-single-gpu3-qualification.source-manifest.json` | `944183b276b231c2dd306dbfc9ddbbf9823d94875e21a94b76008ef2b91ba0a8` |
| `qualify_v11_paired_gpu3.py` | `7940f7aa82320cac5ff9ababe4d806e6a6449575d43d08e6930e2dac4c692ede` |
| `topomem-v11-paired-preregistration.json` | `8da4c6d303f22133e5084f47dd39a0cc970d26f3faaa4b8375a1f93c63764390` |

预注册先在本机生成并通过源清单、用例和顺序校验。执行时由持有 GPU3 的操作者解包源归档并指定私有执行配置。该配置给出远端 worker、解释器和源目录，并使用计划中的设备 UUID 及 `OMP_NUM_THREADS=2`。执行器拒绝复用已有的回执目录。

```text
python -B qualify_v11_paired_gpu3.py run --plan <preregistration.json> --expected-plan-sha256 <plan-sha256> --source-manifest <manifest.json> --controller-root <local-v11-integration> --executor-config <private-executor.json> --audit-directory <new-receipt-directory>
```

## v11 的 GPU3 配对工程校准结果

审计核对了 32 份完整回执及独立进程，运行顺序与预注册逐项相同。v10 和 v11 对应区块的固定输入、程序、选中 KV 字节和输出摘要、执行签名、传输字节、缺页次数与受管 KV 峰值均一致。完整比较见 [v11 工程回执审计](engineering-v11-gpu3-qualification.json)，SHA `6b299a93f8913df2ad2b38a18ef797c352b8086ce279e35e2434ff63b1d93d8e`。

| 用例 | 原生中位数（ms） | LRU 等价（ms） | 直接手写（ms） | 写回（ms） | 原生减写回配对中位数（ms）及胜出块 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 代理 | 47.219 | 51.823 | 47.370 | 45.579 | +1.889，8/8 |
| 对话 | 56.889 | 62.249 | 63.828 | 56.166 | +0.901，7/8 |
| 长文 | 150.981 | 158.675 | 150.554 | 201.936 | −51.567，0/8 |
| Rollout | 21.115 | 23.730 | 19.315 | 22.171 | −1.133，0/8 |
| 汇总 | 276.120 | 296.521 | 280.830 | 325.816 | −50.372，0/8 |

LRU 等价臂比原生慢的汇总配对中位数为 20.154 ms，8 个区块均较慢。直接手写在 rollout 的原生减对照配对中位差为 +1.745 ms，8/8 区块同方向。v10 和 v11 按相同设计区块对比时，LRU 等价臂的 `v10 - v11` 汇总中位差为 +6.428 ms。两个版本的运行发生在不同时间窗口，版本差异仅作描述。该单卡校准使用公开训练用例和手写程序，正式双卡、留出及模型生成能力仍需各自的独立评估。

```powershell
python -B docs/build_v11_gpu3_calibration.py --receipts <v11-raw-receipts> --preregistration <v11-plan> --v10-audit docs/engineering-v10-gpu3-qualification.json --output <new-audit-json>
```
