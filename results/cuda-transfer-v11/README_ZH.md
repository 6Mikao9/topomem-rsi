# 四场景生成器迁移预注册

实验在同一 v11 双卡执行器上比较 v7 原始生成器 `42699e4d5148…` 与 v7 R2 已晋级的 `7d9d37e11e08…`。两版源码逐字节固定。代理是同域引擎迁移，对话、长文和分支回滚是跨场景迁移。`round=2` 为双方共同的预注册入口参数，本实验只执行生成入口。历史 v7 晋级描述的是其当时相对父代的结果，本次没有新增递归改进轮次。

每个场景沿用既有两个训练用例和两个留出用例。四组外部事件、注意力索引、计算位置、拓扑、受管预算与计时定义在各组内固定。代理和长文使用 GPU0/1，对话和分支回滚使用 GPU0/3，均沿用 CPU0 亲和性。应按回执报告生成策略是否实际将 KV 驻留或移动到第二张 GPU。

| 阶段 | 固定规则 | 作业上限 |
| --- | --- | ---: |
| `baseline` | 四场景各跑原生 LRU、六规则初始程序和手写 IR，核对选中 KV 与注意力输出后建立共同训练档案 | 12 GPU |
| `generate` | 两版源码各每场景两槽，使用同一冻结的场景训练上下文。独立 DeepSeek v4.1 Flash 请求，high，单槽上限 65,536 输出 tokens | 16 API，0 GPU |
| `select` | 每槽评估合法程序。按训练可行最低分、平局用槽编号为每源选一份并封存 | 16 GPU |
| `heldout` | 各场景 12 个预注册 Williams 配对块，比较两个生成程序、原生 LRU、直接手写 | 192 GPU |

两版生成器见到相同的可行锚点及排序，模型请求只包含各源码决策选中的训练证据。各候选不会改变其他槽的档案。缺少合法输出、GPU 不可行或中断状态全部保留。若一个生成器没有可行训练结果，留出阶段按预注册的可用臂集合使用对应均衡顺序，缺席对照保持缺失状态。四组最多 16 次 API 调用，提供者输出 token 总上限为 1,048,576，实际 token 用量和完成状态落盘。报告时区分槽数、实际请求数和有效程序数。`request_timeout_seconds=240` 是传输超时设置。

所有留出臂逐块比较实际选中 KV 字节和注意力输出，核对诊断和计时的执行签名、结束时所有权及受管峰值。每次作业均通过同一个冻结 CUDA 评估入口。分数为两个用例总耗时中位数之和，越低越好。严格通过要求完整的 12 个配对增益逐块超过场景阈值。代理为 0.5 ms、对话为 1.5 ms、长文为 3.5 ms、分支回滚为 1.0 ms。另报 v7 等价的逐块正收益及配对中位差阈值。两份生成 IR 相同的对照无法支持生成器迁移增益。全局近似配对 bootstrap 只覆盖晋级源产出相对 seed 源产出、原生和直接手写的四场景共 12 个聚合比较，以 Bonferroni `0.05/12` 校正。每用例仍报告所有配对数据。测试没有中间验证环节，留出结果不用于重选程序。

留出数据沿用原先工件，若已进入 v7 留出评估则标为相同用例的复测。模型输入、生成槽及选择证据可按阶段 JSON 与 SQLite 日志重建。此试验使用文本派生的 CUDA 分块稀疏注意力和实际 KV 生命周期，证据范围限于这些固定合成场景。真实服务另有独立评估。

## 冻结工件和启动

| 工件 | SHA-256 |
| --- | --- |
| `../topomem_v11_transfer.py` | `41fbadf5611e78fa6d41b257ffc1c13acb360018697d84e5ed5e54bae537cf43` |
| `preregistration.json` | `649c8d0d8912db3a69f0d0363de078a9dab7685d129a2af1586aafd098ee643e` |
| `../topomem-v11-integration.source-manifest.json` | `877873960cd4da3c332891495fdc86d30fa59725b879d0fcffccb81fdef0c767` |
| `../topomem-v11-integration.source.zip` | `1ed35df6bb3a05407205a5f10d40adefa4cb5f0499c30dc82ef80c3ef4c1212f` |

`prepare` 已在本机执行。以下命令使用操作者提供的私有执行配置，其根对象以四个场景为键，沿用原 `executor` 字段，并把 `source` 和 `worker` 定向到正式双卡 v11 源包。`baseline`、`select`、`heldout` 需要 GPU。`generate` 可在服务占用 GPU 时运行，从环境变量或显式密钥文件读取现有授权密钥。配置和密钥只留在操作者环境。

```powershell
$scratch = 'D:/ws/.scratch'
$driver = "$scratch/topomem_v11_transfer.py"
$shared = @('--study', "$scratch/topomem-v11-generator-transfer",
  '--source-root', "$scratch/topomem-v11-integration",
  '--source-manifest', "$scratch/topomem-v11-integration.source-manifest.json",
  '--expected-plan-sha256', '649c8d0d8912db3a69f0d0363de078a9dab7685d129a2af1586aafd098ee643e')
python -B $driver baseline @shared --executor-config <private-executor-config>
python -B $driver generate @shared
python -B $driver select @shared --executor-config <private-executor-config>
python -B $driver heldout @shared --executor-config <private-executor-config> --expected-selection-sha256 <selection-digest>
```

本地 4 项测试通过。阶段测试使用替身执行器完成 12 次基线、16 个候选请求、16 次候选评估和 192 次留出提交，并验证中断效果无重发、留出前必须封存、错误选中字节使门槛失败。另用真实 v11 单卡工程回执验证诊断读取器。迁移试验真实回执应单独记录。
