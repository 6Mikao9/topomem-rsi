# TopoMem RSI 中文说明

[English](README.md) · [两项任务与递归链](docs/ARCHITECTURE_ZH.md) · [自主迭代与人机分工](docs/AUTONOMY_BOUNDARY_ZH.md) · [v4.1 选择协议](docs/V4_SELECTION_PROTOCOL_ZH.md) · [显存程序契约](docs/MEMORY_PROGRAM_CONTRACT_ZH.md) · [SGLang 复现指南](docs/REPRODUCE_SGLANG_ZH.md) · [研究框架复核](docs/RSI_FRAMEWORK_REEVALUATION_20261110_ZH.md)

## 从这里开始

1. **打开离线演示。** 下载 HTML 后在启用 JavaScript 的本地浏览器打开。[最新 v4.1 两轮证据](demo/topomem-v41-final.html)展示模型提出的改进、验证裁决、源码和四场景留出；[此前多卡与 SGLang 证据](demo/topomem-evidence.html)保留原始实验。两者硬件配置与实验批次分别说明。
2. **做本地 CPU 检查。** 在仓库根目录用 Python 3.10 或更新版本运行。以下测试只用标准库，可以直接从源码运行。

   ```powershell
   python -B -m unittest discover -s tests -p 'test_memory_program.py'
   python -B -m unittest discover -s tests -p 'test_topology_report*.py'
   ```

3. **离线审计归档。** 下面的命令校验工程 ZIP 和原始成员摘要，无需解包，返回四个归档各自的 `verified_members`。审计页也给出从回执重建摘要的流程。 [工程归档与审计](results/engineering-calibration/README.md) · [SGLang 手写对照复算](results/sglang-hand-sweep-v2/REPRODUCE_ZH.md)

   ```powershell
   python -B results/engineering-calibration/extract_receipts.py --manifest results/engineering-calibration/receipts-manifest.json
   ```

需要自行启动 GPU 作业时，按[本地 CUDA 配置](#准备本地配置)或[SGLang 复现指南](docs/REPRODUCE_SGLANG_ZH.md)部署。SGLang 演示使用随机初始化后经玩具文本微调的 tiny GLM 架构验证 checkpoint，观察到 EOS 输出。测量覆盖固定请求回放和 KV 正确性。生产吞吐及应用答题质量需要独立测量。 [模型来源](results/sglang-qualification/model-provenance/UPSTREAM_MODEL_CARD.md)

## 两项任务

1. **生成显存管理程序。** 给定稀疏注意力、实际 GPU 拓扑和固定场景，模型生成 `MemoryProgram`。规则可选择 KV 放置、副本、迁移、压力降级、预取、驱逐和回收。受信执行器校验事件、动作、特征、路线、容量和引用，再在原生、手写和生成臂上测量。
2. **改进程序生成器。** 生成器有 `build_prompt` 和 `build_meta_prompt` 两个入口。前者选择构造、变异或交叉，并指定父代、证据及规则。后者选择修订策略和训练证据。同一次元模型响应返回 `source` 和 `plan`。评估子代前，控制器绑定源码摘要并封存方案中的观察、机制改动、预期和证伪条件。控制器用同起点候选与配对验证决定晋级，记录源码差异和下一轮实际任务及元调用。方案中的预期是待测假设。

目标合同固定注意力实现和选集、物理 GPU 与有向连接、计算 rank、显存容量、请求生命周期、外部调度、批次和安全点。模型只接触训练范围的目标描述、候选档案和反馈。裁判保管验证及留出信息，决定可行性和分值。v4.1 用重复训练测量分别选择发现的管理器、初始或晋级源码的实际产物、固定生成器产物；旧版本的封存规则保留在各自原始清单中。

## 当前证据

配对区块在独立作业中重复执行各冻结比较臂。生成器运行次数和源码晋级次数单列。

DeepSeek 当前可生成显存 IR，并修改 `build_prompt` 与 `build_meta_prompt` 两个生成入口。现有训练档案保存已测程序、来源和反馈。完整框架代码、显存后端和跨运行语义知识库的自主迭代仍待实现。框架在现有能力范围内负责提出和评估显存改进。人主要设定边界、复核测试并反馈可重现的问题。v12 和 v13 后端改动均属人工工程。[能力边界与下一阶段验收](docs/AUTONOMY_BOUNDARY_ZH.md)

### 最新 v4.1 GPU3 补充实验

两轮共 20 次 DeepSeek 调用、134 份 GPU 回执。模型提出显存 IR 和两个生成入口的改动，人工没有改写这些候选。两轮源码均未晋级，晋级源码的后续实际复用为 0。R0 子代相对本轮父代慢 22.644 ms；R1 子代虽比历史管理器快 9.676 ms，却比本轮父代新产物慢 2.253 ms，因此仍拒绝。发现的管理器与晋级源码产物分开保留。

| 留出产物 | 对原生胜出／中位收益 | 对固定生成器胜出／中位收益 | 对手写策略胜出／中位收益 |
| --- | --- | --- | --- |
| 管理器发现结果 | 0/6，−37.739 ms | 0/6，−15.536 ms | 6/6，+35.415 ms |
| 初始或已晋级源码产物（本次来自初始源码） | 0/6，−36.516 ms | 0/6，−15.477 ms | 6/6，+36.100 ms |

每块分值为四场景耗时中位数之和，收益为对照减候选。四案实际都在 GPU3 执行 H64 注意力；使用既有公开语料、128 KiB 受管 GPU KV 和 32 MiB pinned host。它是单卡工程复评，不替代原多卡结果，不测 LongBench 答题质量。这次没有证明 RSI 优势。所有比较臂使用同一人工 v13 执行器，其工程收益另行归因。

[原始报告与复核命令](results/cuda-rsi-v41-gpu3/README_ZH.md) · [独立训练及留出审计](results/operations/cuda-rsi-v41-gpu3-independent-audit/README_ZH.md) · [219 作业服务器归档](results/operations/followup-server-evidence/README_ZH.md)。重建最新演示可运行 `python -B -m rsi_memory.demo_v4 --report results/cuda-rsi-v41-gpu3/controller/report.json --harness-directory results/cuda-rsi-v41-gpu3/controller/harnesses --output demo/rebuilt-v41.html`。历史运行必须使用其归档冻结源码；当前 main 另含人工持久化和终局状态修复，不可混用源码恢复旧运行。

### SGLang v3 正式实验

两轮均未晋级生成器源码。递归臂的封存程序来自 R0 的初始生成器，使用 4096 热槽，并将两个 rank 各自第 0 层的冷 KV 放在本卡 HBM。正式留出包含 12 个配对区块，每个区块有两个用例。收益取同区块的对照耗时减递归臂耗时。门槛要求每块收益为正，且中位收益严格超过 11.492 ms。

| 对照 | 递归臂更快 | 配对中位收益 | 预注册裁决 |
| --- | ---: | ---: | --- |
| 本实验原生 4096 槽对照 | 7/12 | +10.958 ms | 未通过 |
| 固定生成器 | 0/12 | −32.569 ms | 未通过 |
| 训练选优的手写策略 | 0/12 | −26.065 ms | 未通过 |

固定生成器的封存程序选择了 `resident-6144`，与手写策略的回执所记内存配置相同。同批区块的描述性比较中，固定生成器方案相对本实验原生 4096 槽对照为 12/12 更快，配对中位收益为 +35.585 ms。该程序由未改写的初始生成器产出。 [独立终局审计](results/sglang-rsi-v3/independent-audit/TERMINAL_AUDIT_ZH.md) · [正式控制器报告](results/sglang-rsi-v3/controller/report.json) · [封存程序来源](results/sglang-rsi-v3/selected/selection.json)

预注册追加对照将同一封存程序与原生 6144 槽全主机后备配对。封存程序 5/8 区块更快，配对中位收益 +11.923 ms，未通过全区块门槛。比较同时涉及 4096 与 6144 槽热容量和后备位置。 [追加对照审计](results/sglang-native6144-ablation-v1/README_ZH.md)

### 早期实验

| 实验 | 已观察到的状态 |
| --- | --- |
| producer 保留 v13 人工工程对照 | 在静态约束证明原位保留是唯一合法候选时，省去通用候选求值。同一 LRU 等价 IR 的动作、生命周期、KV、输出和链路字节一致，四案合计较 v11 为 8/8 更快、配对中位改善 +11.661 ms；相对原生仍为 0/8 胜、中位慢 11.449 ms。这是人工执行器优化，不是模型提出的显存策略或 RSI 收益。[工程证据](results/engineering-producer-v13/README_ZH.md) |
| v4.1 H256 双卡中断实验 | 四案按固定映射使用 GPU0/1，宿主机重启中断了首轮训练矩阵。保留 26 份新回执、10 次完成模型调用及未知作业；没有结算轮次、源码晋级或留出结果，未拼接补测。[中断与恢复证据](results/cuda-v41-reboot-interruption/README_ZH.md) |
| 动态特征 v12 工程对照 | 同一 writeback IR、同一 GPU3 上执行 8 个配对区块共 24 个作业。动作、KV、输出及链路字节一致，新按需特征计算在合计耗时上 0/8 胜，配对中位慢 9.900 ms；长文单案慢 9.397 ms。未合入默认 GPU 运行时，不计为 RSI。[失败工程证据](results/engineering-dynamic-v12/README_ZH.md) |
| v4 原提示试跑 | 审查发现继承的元提示与 fresh-output 裁判不一致，已停止。6 次完成模型调用、1 次未知调用和 11 份 GPU 回执保留；0 个结算轮次，无封存或测试。v4.1 修正实际出站消息，旧试跑不重试、不混入新版结果。[停止记录](results/prompt-v4-abort/README_ZH.md) |
| CUDA v6 Agent | 结算 R0 后停止。子代两个 AST 均变化，同起点算子、父代、证据和元模式相同，未晋级。验证 12 个区块中子代均慢于项目的直写/LRU 参考。区块分值中位数为子代 146.529 ms、参考臂 120.898 ms。五个实测臂的第二张 GPU 管理 KV 峰值均为 0 B。审计复核了 66 份回执，保留 R1 部分调用和未知状态。 |
| CUDA v7 Agent | 六轮 DeepSeek 生成和 12 个留出区块均已结算。唯一晋级的是 R2 生成器源码，其内存程序恢复已观察到的历史最佳行为，相对当轮父代 12/12 胜、配对中位改善 +126.658 ms，相对项目直写/LRU 参考 0/12 胜。R3–R5 的九次实际任务与元调用复用了晋级源码。封存的递归臂内存程序来自 R0 初始生成器，留出对直写/LRU 参考 0/12 胜、配对中位改善 −29.172 ms，对固定生成器 6/12 胜、配对中位改善 −0.142 ms。 |
| CUDA v11 四场景迁移 | 对比冻结的初始生成器与 v7 R2 晋级生成器。192 份留出作业通过所记录的选中 KV、注意力输出和所有权检查。晋级源产出对项目直写/LRU 参考 48/48 区块较慢。对初始源，Chatbot 虽然 12/12 较快，但只有 11 个区块超过预登记的 1.5 ms 门槛。四场景均未通过严格迁移门槛。全部四臂在第二张 GPU 的受管 KV 峰值为 0 B。[迁移审计](results/cuda-transfer-v11/audit/AUDIT_ZH.md) |
| SGLang resident v2 原生资格 | 用最终 v2 服务源码在固定公开回放上执行六次双卡 TP2／attention DP2 原生作业。4096、6144 和 8192 槽各两次，双 rank 选中 packed KV、输出 IDs、实际派发和 132 个活动全局 forward tick 均一致。适配器支持启动热容量、静态冷层放置及经过逐案证明的全驻留。[原生资格审计](results/sglang-qualification/ingress-resident-v2/AUDIT_ZH.md) · [部署与协议](docs/SG_RESIDENT_V2_ZH.md) |
| SGLang 九策略手写训练选优 | 单个正式训练用例上的 38 个作业全部结算，其中三个区块共 27 个计时作业。手写 `resident-6144` 获选，相对原生 6144 槽和本实验原生 4096 槽对照的配对中位收益分别为 +22.234 ms 和 +25.187 ms。原生 4096 槽训练窗口的极差给出 11.492 ms 的正式门槛。正式 v2 搜索因通用提示强制 `allocate/place` 而全驻留模式禁止该规则的契约冲突停止。该训练选优为正式比较提供手写对照。[训练审计](results/sglang-hand-sweep-v2/README_ZH.md) · [原始选优](results/sglang-hand-sweep-v2/sweep/result.json) · [训练用例绑定](results/sglang-hand-sweep-v2/training-case/manifest.json) |
| 独立 resident v3 工程探针 | 固定公开 tiny GLM 回放的两组原生／全驻留配对中，人工全驻留原语分别缩短 28.003 和 28.479 ms。每 rank 冷主机 payload 由 129,226,752 B 降至 0 B，新增 CUDA 占位逻辑 3,936 B。这两次观察用于决定把该原语纳入九策略手写对照。checkpoint 随机初始化后经玩具文本微调，回放输出为 EOS。[工程探针审计](results/sglang-qualification/resident-v3/README.md) |
| Stock SGLang 独立探针 | TP2 主干的 attention DP2 验证了各 attention rank 的真实派发，每 rank 八个本地 query heads。独立 UnifiedRadixCache 探针验证两分支共享 1,536 个物理前缀行，并各保留 128 个互不重叠的后缀行。该 radix fixture 的缓存与隔离推理输出 IDs 均为 EOS。 |

生成器源码的晋级和后续复用属于源码任务。封存程序的留出结果属于显存管理任务。[v6 首轮审计](results/cuda-rsi-v6/agent_tools/AUDIT_R0_ZH.md) · [v7 六轮与留出审计](results/cuda-rsi-v7/agent_tools/AUDIT_PROGRESS_ZH.md) · [SGLang 资格流程](integrations/topology_memory/SGLANG.md) · [SGLang 探针证据](results/sglang-qualification/evidence_manifest.json) · [SGLang resident v2 手写对照与预算](docs/SG_RESIDENT_V2_ZH.md)

## 准备本地配置

下面是可选的实机流程。控制器用 Python 3.10 或更新版本，从仓库根目录运行时可省略安装。CUDA 裁判主机使用支持目标设备的 PyTorch。SGLang 使用独立的固定依赖。 [SGLang 复现指南](docs/REPRODUCE_SGLANG_ZH.md)

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

构建可重复的 CUDA 部署源码包。

```powershell
python -B -m rsi_memory.source_bundle --output deploy/cuda-source.zip
```

1. 在裁判主机运行 `python -B -m integrations.topology_memory.torch_profile --output link-profile.json`，保留两张实际 GPU 各自的 pinned host H2D、D2H 原始测量、UUID 和容量。测量文件用于构建真实有向拓扑。副本容量和路线在实验开始前固定。
2. 将部署包解压到裁判工作目录下的新源码目录。`SOURCE_MANIFEST.json` 的 `files` 字段将相对文件路径映射到字节 SHA-256，覆盖裁判及其全部可导入文件。把打包命令返回的 `source_manifest_sha256` 固定在实验输入中。执行器通过现有 OpenSSH 配置连接，并在运行前检查源文件、设备身份及 GPU 占用。
3. 准备本地 `private-build-inputs.json`，按下表提供构建器输入。目标配置、主机地址、设备标识和凭据保存在自己的实验目录。控制器可从本机 `OPENCODE_GO_API_KEY` 或通过 `--key-file` 读取模型凭据。

| 输入字段 | 内容 |
| --- | --- |
| `experiment_id`、`output_root` | 独立实验名和本地结果根目录。下面的命令使用 `runs`。 |
| `profiles` | 四种场景分别提供 `topology`、`target`、`compute_node`、`device_map`、`compile`。`compile` 固定页大小、维度、层数、top-k 和文字到行的转换参数。 |
| `executor` | `host`、`worker`、`root`、`python`、`source`、`gpu_indices`、`gpu_uuids`。可选 `environment` 仅接受执行器声明的配置项。 |
| `calibration`、`source_manifest_sha256` | 指向有向链路测量文件并提供固定源码清单的 SHA-256。 |
| `initial_program`、`manual_program` | 两个完整 `MemoryProgram` 对象。可读入仓库中的原生 LRU 等价策略和手写策略作为起点与对照。 |
| `workload_directory` | 四个主 fixture 与 `variants/` 的目录。示例为 `examples/workloads`。 |
| `paired_blocks`、`warmup`、`samples`、`timeout` | 分别固定每个验证和留出集的区块数、每臂预热和计时样本数、工作进程超时。 |

构建器校验 GPU 容量、固定计算节点和每条 pinned host 连接是否匹配实际 profile。它为每个场景分别写训练、验证和留出配置。训练可用同一自编内容的两个不同固定种子，验证和测试使用不同内容的变体。[构建器](rsi_memory/experiment_cases.py) · [手写与 LRU 对照](examples/policies/README.md)

```powershell
python -B -m rsi_memory.experiment_cases `
  --inputs private-build-inputs.json --output sealed-configs
```

## 运行与留出

选择一份封存场景配置和新的控制器目录。两臂从同一个种子开始。递归臂每轮给活动父代和子代同量候选槽。固定臂保留原生成器，获得同样的候选和调用预算，其元调用产物弃用。台账分开记录预算上限、已知实际用量和未知状态。

```powershell
$RunArgs = @(
  '--output', 'runs/agent_tools/controller',
  '--adapter', 'rsi_memory.cuda_domain:factory',
  '--adapter-config', 'sealed-configs/agent_tools.json',
  '--model', 'deepseek-v4.1-flash', '--rounds', '6', '--slots', '2',
  '--max-output-tokens', '65536', '--min-paired-blocks', '12',
  '--min-improvement', '0.5', '--wall-seconds', '43200'
)
python -B -m rsi_memory.topology_rsi run @RunArgs
```

只有 `run` 调用模型，可从本机环境变量或 `--key-file` 读取凭据。远端裁判收到固定输入和程序。发现阶段结算后检查失败、待完成和晋级记录。恢复使用相同源码、模型和实验合同，未知调用原样留档。确认发现阶段结束，再用同一组参数封存并打开独立留出评估。`freeze` 在本地执行，`test` 需要 GPU 并禁用模型 API。

```powershell
python -B -m rsi_memory.topology_rsi freeze @RunArgs
python -B -m rsi_memory.topology_rsi test @RunArgs
```

`journal.sqlite` 记录预登记方案、调用、训练裁判、配对区块和版本决定。`frozen.json` 绑定主策略和固定对照。留出结论读取 `pair/frozen/test`、对照差值和实际回执。晋级需要两入口在当轮同起点作出结构变化，且父子配对每区块方向一致，改善中位数超过预设门槛。下一轮复用需要实际任务和元请求都绑定晋级源码。运行时可行性、生成器晋级和相对原生的性能收益是独立证据。

## 离线报告

在仓库根目录运行以下 CPU 命令，用六轮原始台账、已封存四场景迁移、SGLang 手写训练选优和独立工程审计重建离线视图。请选择新的输出文件名，已有 HTML 会保留。

```powershell
python -B -m rsi_memory.topology_report `
  --run Agent=results/cuda-rsi-v7/agent_tools/controller `
  --transfer results/cuda-transfer-v11 `
  --manual-sweep results/sglang-hand-sweep-v2/sweep `
  --engineering results/engineering-calibration/engineering-v11-gpu3-qualification.json `
  --engineering results/sglang-qualification/resident-v3/audit.json `
  --output demo/rebuilt-archive.html
```

要查看实际布局和作业证据，按[证据说明](results/EVIDENCE.md)从归档恢复原始 GPU 回执。

```powershell
python -B -m rsi_memory.topology_report `
  --run 'Agent=runs/agent_tools/controller' --receipts runs/agent_tools/receipts `
  --scenario examples/workloads/agent-water-inspection.json `
  --scenario examples/workloads/long-document-estuary-qa.json `
  --scenario examples/workloads/chatbot-trip-planning.json `
  --scenario examples/workloads/rollout-bridge-options.json `
  --output agent-evidence.html
```

打开 HTML，可切换运行、迭代和场景，查看候选程序、规则执行、评估前方案、源码差异和失败记录。需要显示拓扑时传 `--topology`。可用 `--scenario 类型=场景绑定.json` 按摘要核对实际 `MemoryScenario`。`--transfer` 读取独立迁移研究目录，`--manual-sweep` 读取手写训练选优，`--engineering` 读取工程审计 JSON。无回执的场景显示输入状态。报告生成过程只读取已有台账和回执。[查看示例](examples/topology-report/README.md)

## 场景与集成范围

| 固定场景 | CUDA 裁判读取的机制证据 |
| --- | --- |
| Agent | 固定工具调用与返回事件、会话前缀和选中 KV。 |
| LongBench 风格长文问答 | 自编长文和问题、文本派生 KV、长上下文选集与输出核验。 |
| Chatbot | 多轮前缀保留与逐轮追加。 |
| Rollout | 分支共享页、物理写时复制、分支后缀及退出后的引用释放。 |

CUDA 裁判按固定外部逻辑时钟执行真实 KV 和稀疏注意力张量运算。`native_lru` 是本项目的直写/LRU 参考。decode 追加会物化不可变 KV 块版本，相应整块操作成本进入该执行器的计时。v7 分值是各场景耗时中位数之和，单位 ms，计入强制回收。预分配及编译在计时外。管理 KV 峰值按节点统计，PyTorch 分配峰值另列。声称使用第二张 GPU 时需出示该节点实际 KV 占用或链路传输回执。

SGLang 适配是独立模型服务路径。程序用 `configure/hot_capacity` 选择已测的原生热容量。分层后备模式用 `allocate/place` 选择冷 KV 层放置。程序用 `configure/backing_mode` 选择全驻留后，由受信编译器为每个用例验证容量。全驻留原语由人工实现。运行绑定模型、选择器、请求、计算 rank 及实测链路，后端继续管理原生热缓存。不同后端的性能、正确性和场景数据分别呈现。[场景契约](examples/workloads/README.md) · [SGLang 适配](integrations/topology_memory/SGLANG.md) · [模型来源与配置](docs/SG_CONFIGURABLE_RELEASE.md) · [证据方法](docs/RSI_EXECUTION_EVIDENCE_ZH.md)
