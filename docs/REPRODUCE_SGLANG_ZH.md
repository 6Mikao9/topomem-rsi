# SGLang 双卡实验复核与复现

该路径使用 SGLang 0.5.20、TP2 主干和 attention DP2。两个固定 rank 分别执行自己的请求，使用 `torch-unfused` 选择器、原生 DSA attention 和 656 B packed KV。公开实验使用随机初始化后经玩具文本微调的 tiny GLM 架构验证模型，实测输出为 EOS IDs。指标是固定请求组预热后的整窗耗时，正确性覆盖选中 KV、输出 ID、实际派发和全局 forward。应用质量和生产吞吐需要另测。 [模型来源](../results/sglang-qualification/model-provenance/UPSTREAM_MODEL_CARD.md) · [实测训练拓扑与场景承诺](../results/sglang-hand-sweep-v2/training-case/manifest.json)

复核归档只需本地 Python 3.10 或更新版本。选择从源码运行或按 README 创建虚拟环境后做可编辑安装。自行部署实机需要 Linux GPU 环境，启动模型发现还需要控制器本机凭据。 [CPU 检查与归档演示](../README.zh-CN.md#从这里开始)

## 1. 先复核已归档的证据

| 证据 | 可复核内容 |
| --- | --- |
| [Resident v2 原生资格](../results/sglang-qualification/ingress-resident-v2/AUDIT_ZH.md) | 同一 v2 worker 的六次原生校准。4,096、6,144、8,192 槽各两次，逐 rank packed KV、输出和实际全局批次一致。 |
| [九策略训练选优](../results/sglang-hand-sweep-v2/README_ZH.md) | 单个训练用例、38 份原始回执、三个计时区块。`resident-6144` 获选为手写对照。 |
| [训练选优复算](../results/sglang-hand-sweep-v2/REPRODUCE_ZH.md) | 校验工件与回执 ZIP 成员摘要，重算训练选优及 11.492290 ms 的正式门槛。 |
| [独立工程探针](../results/sglang-qualification/resident-v3/README.md) | 人工实现全驻留原语在一个公开用例上的两组配对。与正式搜索及正式留出分开。 |
| [正式 v3 控制器报告](../results/sglang-rsi-v3/controller/report.json) | 两轮生成与验证、冻结来源和 12 个留出配对区块。 |
| [冻结后原生 6144 追加对照](../results/sglang-native6144-ablation-v1/README_ZH.md) | 同一冻结程序的 8 个区块与 32 份计时回执。 |

正式 v2 尝试因生成提示要求 `allocate/place`，而全驻留程序必须省略冷放置规则而停止。v3 只修订本地控制器发给模型的完整程序语法，沿用已核验的 v2 worker、容量 profile 和训练对照证据。重新开始时使用新的实验 ID、控制器目录和回执目录。 [v2 协议及停止原因](SG_RESIDENT_V2_ZH.md)

在仓库根目录运行下列命令，可直接从已归档原始回执重建工程探针摘要。命令只写新的输出文件，复算的是公开用例的两组工程对照。九策略训练选优的独立复算见上表。

```sh
python -B results/sglang-qualification/resident-v3/build_audit.py --evidence results/sglang-qualification/resident-v3/resident-v3-evidence.zip --output resident-v3-audit-local.json
```

## 2. 在操作方主机绑定输入

正式归档提供原始四案请求、目标、拓扑、外部调度、链路证据及分词承诺。一训练、一验证、两留出用例与正式控制器绑定。原运行的私有 launcher、连接信息和 schema 配置由操作方保存。复现实机结果时使用匹配的部署，迁移到新设备或模型时重新绑定。 [四案使用说明](../results/sglang-rsi-v3/formal-cases/README_ZH.md) · [四案 ZIP](../results/sglang-rsi-v3/formal-cases/formal-four-cases.zip) · [公开成员摘要](../results/sglang-rsi-v3/formal-cases/PUBLIC_CASE_OVERLAY.json)

1. 准备可运行的 Linux SGLang 环境、目标模型、CUDA 和隔离 NCCL。绑定两张实际 GPU 的 UUID、计算 ordinal、设备能力和有向 pinned-host 链路证据。历史目标和模型由内容摘要固定。`topomem_sglang_rsi.make_config` 及远端裁判会核对这些绑定。 [SGLang 适配](../integrations/topology_memory/SGLANG.md)
2. 将 [v2 worker ZIP](../results/sglang-qualification/ingress-resident-v2/source/resident-v2-worker.zip) 放入一个新的远端实验根的 source 子目录。远端部署需要相邻的可信常规文件 `sglang_launch.json` 和可执行的 `sglang_launch.py`。launch schema 2 由操作方填写 `python`、`model`、`nccl_isolate`、`cuda_home`、`cache_root`、`gpu_indices`、`gpu_uuids`。发布包中的历史 launch 路径只用于核对原运行。新部署填写自己的路径并重新封存实际 launcher 与全部源文件。历史 `SOURCE_MANIFEST.json` 的字节 SHA-256 为 `b73995aea418432831366a7c2de0af038fbe8eae2f2f85ac164d2e88e4c81909`。
3. 使用归档解开的 `formal-cases/cases/` 作为本地 case 根，并在远端实验根部署相同的 `cases/` 子树。`CASE_SET` 和 `TOKENIZATION_RECEIPT` 分别指向其中的 `formal_case_set_manifest.json` 与 `formal_tokenization_receipt.json`。保留同目录的父清单 `case_set_manifest.json` 供适配器核对预登记索引。确保模型、请求输入、外部调度与分词承诺一致，生成器的运行输入只接触训练拆分。 [Case 绑定](SG_GATED_QUALIFICATION_ZH.md)
4. 对实际源码、运行环境、设备与用例执行原生 A/A。校准 profile 绑定 11 个实际 serving adapter 文件和完整运行时身份。源文件、launcher、模型、拓扑、调度或 manifest 变化后，按新绑定重新资格和校准，再建立同源训练选优与正式实验。历史校准与运行命令见 [原生校准复现](../results/sglang-qualification/ingress-resident-v2/REPRODUCE_ZH.md)。

本地解开 v3 的 `frozen-source.zip`，也就是 `contract-v3-rc.source.zip` 原件。此源码 ZIP SHA-256 为 `ab8479bbb8b017a50673c07485ddb0fd88252dc24ec9a7467e4b214f9dfdb7b9`，其中 `LOCAL_SOURCE_MANIFEST.json` 字节 SHA-256 为 `f452f06968cb2b1df687eba72e6dabcd3e5e419fe15bd205b555dce1f0485c9b`。在 Bash 中将 `CONTROLLER_SOURCE_ZIP` 设为源码包路径，将 `NEW_CONTROLLER_SOURCE_DIR` 设为待创建的解包目录。后续命令在该源码根执行。两份清单分别绑定本地控制器和远端 worker。模型 API key 留在本地控制器，只有 `run` 读取它。SSH 启动的 worker 接收已封存输入、程序、资格回执标识及预算。 [冻结源码 ZIP](../results/sglang-rsi-v3/frozen-source.zip)

```sh
printf '%s  %s\n' ab8479bbb8b017a50673c07485ddb0fd88252dc24ec9a7467e4b214f9dfdb7b9 "$CONTROLLER_SOURCE_ZIP" | sha256sum --check
python -m zipfile -e "$CONTROLLER_SOURCE_ZIP" "$NEW_CONTROLLER_SOURCE_DIR"
cd "$NEW_CONTROLLER_SOURCE_DIR"
python -B -m topomem_sglang_rsi.contract_v3 --help
```

## 3. 绑定训练对照和正式协议

在本地用自己的路径填入以下 Bash 变量。进入 v3 源码目录前，可先将输入路径设为绝对路径，便于之后核对它们的位置。

| 变量 | 输入位置 |
| --- | --- |
| `CASE_SET`、`TOKENIZATION_RECEIPT` | 同一份本地已封存 case 包内的 `formal_case_set_manifest.json` 和 `formal_tokenization_receipt.json`。case 子目录与清单位于同一父目录。 |
| `HOT_PROFILE`、`INITIAL_PROGRAM` | 实机校准产生的 `hot_metadata_profile.json`，以及封存的起始 `MemoryProgram` JSON。历史参数可从[公开 schema 2 模板](../results/sglang-hand-sweep-v2/schema2.template.json)的 `initial_program` 字段取出。 |
| `LAUNCH_CONFIG`、`WORKER_SOURCE_MANIFEST` | 本地持有的实际 `sglang_launch.json` 和部署包 `SOURCE_MANIFEST.json`，字节与远端一致。 |
| `SSH_HOST`、`REMOTE_ROOT`、`REMOTE_SOURCE` | 本机 SSH 别名、远端专用实验根及其下的源码子目录名。 |
| `REMOTE_WORKER`、`REMOTE_LAUNCHER` | 远端 source 中 `sglang_executor_worker.py` 和 `sglang_launch.py` 的路径。 |
| `SCHEMA2_CONFIG`、`SCHEMA3_CONFIG`、`SWEEP_RECEIPTS`、`FORMAL_RECEIPTS`、`CONTROLLER_DIR` | 本地分别存放训练配置、正式配置、两组原始回执及控制器台账的路径。`SWEEP_EXPERIMENT_ID` 和 `FORMAL_EXPERIMENT_ID` 分别为训练和正式实验 ID。 |
| `COMPLETED_SWEEP`、`LOCAL_KEY_FILE` | 已结算训练扫参目录和本地密钥文件路径。用 `OPENCODE_GO_API_KEY` 环境变量时，`run` 可省略 `--key-file`。 |

```bash
case_args=(--case-set "$CASE_SET" --tokenization-receipt "$TOKENIZATION_RECEIPT"
           --hot-profile "$HOT_PROFILE" --initial-program "$INITIAL_PROGRAM"
           --launch-config "$LAUNCH_CONFIG" --source-manifest "$WORKER_SOURCE_MANIFEST"
           --host "$SSH_HOST" --remote-root "$REMOTE_ROOT" --remote-source "$REMOTE_SOURCE"
           --remote-worker "$REMOTE_WORKER" --remote-launcher "$REMOTE_LAUNCHER")

python -B -m topomem_sglang_rsi.make_config "${case_args[@]}" \
  --experiment-id "$SWEEP_EXPERIMENT_ID" --receipt-directory "$SWEEP_RECEIPTS" \
  --output "$SCHEMA2_CONFIG"
```

用 schema 2 生成九策略训练预检，核对候选、容量、顺序和 `sweep_sha256`，然后向新的输出目录执行同一已核对计划。完整手写训练命令和回执复核见 [训练选优复现](../results/sglang-hand-sweep-v2/REPRODUCE_ZH.md)。如复用历史训练选优，保留原 schema 2 配置和 38 份完整回执，并通过本地离线重放。替换部署路径或源摘要会产生新的输入身份。

```bash
python -B -m topomem_sglang_rsi.make_config "${case_args[@]}" \
  --experiment-id "$FORMAL_EXPERIMENT_ID" --receipt-directory "$FORMAL_RECEIPTS" \
  --manual-sweep "$COMPLETED_SWEEP" --manual-sweep-config "$SCHEMA2_CONFIG" \
  --validation-blocks 6 --test-blocks 12 --output "$SCHEMA3_CONFIG"

formal_args=(--adapter-config "$SCHEMA3_CONFIG" --output "$CONTROLLER_DIR")
python -B -m topomem_sglang_rsi.contract_v3 run "${formal_args[@]}" --key-file "$LOCAL_KEY_FILE"
python -B -m topomem_sglang_rsi.contract_v3 freeze "${formal_args[@]}"
python -B -m topomem_sglang_rsi.contract_v3 test "${formal_args[@]}"
```

先等待 `run` 结算，检查调用、失败和晋级记录，再运行 `freeze`。`test` 在封存后单独打开留出。`freeze` 和 `test` 均禁用模型 API，`test` 会执行实际 GPU 评估。正式协议固定两轮、每个活动父代和子代各两个候选槽、固定生成器两组相同槽数、每臂每轮最多五次模型调用、每次 65,536 输出 tokens、高推理强度、六个配对验证区块和十二个留出区块。两臂各占一个元调用槽，固定臂弃用元产物。门槛是 5 ms 和本实验原生 4096 槽对照三次训练窗口极差两倍中的较大值。配对区块是固定程序的重复执行，独立的生成器运行次数单独报告。 [严格协议与源码谱系](ARCHITECTURE_ZH.md)

## 4. 区分生成源码与生成的显存程序

- **生成器源码**含 `build_prompt` 和 `build_meta_prompt`。元请求携带活动源码和所选训练证据，一次响应提交候选 `source` 和改进 `plan`。评估前台账固定源码摘要、观察、机制改动、预期和证伪条件。晋级后实际任务与元调用的源码摘要用于核对复用。
- **显存程序**是 `MemoryProgram` JSON。v3 的分层模式要求 `allocate/place`。全驻留模式要求选择 6144 热槽和 `configure/backing_mode`，并省略全部 `allocate/place`。`mutate` 一次只能替换一条现有规则，从完整分层策略切换到全驻留通常需使用 `construct` 生成完整程序。全驻留原语、编译器、容量、调度和裁判由受信运行时提供。

正式 `test` 结算后保留 `journal.sqlite`、`frozen.json`、`report.json` 和回执。使用正式发布附带的 `export_sglang_controller_results.py` 和终态 `report.json` 的字节摘要生成公开导出。该工具需要操作方本地原配置来核对回执和版本。 [计划、候选与裁判合同](MEMORY_PROGRAM_CONTRACT_ZH.md)

```bash
terminal_sha=$(sha256sum "$FORMAL_RUN_ROOT/controller/report.json" | cut -d ' ' -f 1)
python -B "$RESULT_EXPORTER" --run-root "$FORMAL_RUN_ROOT" \
  --private-config "$SCHEMA3_CONFIG" --source-archive "$CONTROLLER_SOURCE_ZIP" \
  --hand-audit "$HAND_AUDIT" --terminal-sha256 "$terminal_sha" --output "$NEW_EXPORT_DIR"
```

`selected/adaptive.program.json` 和 `selected/fixed.program.json` 是两个封存程序。`selected/selection.json` 记录来源。`selected/active.generator.py` 是终轮活动生成器。候选对应的生成器源码按实际生成调用另存，初始测量锚点没有对应模型生成调用。活动源码与封存程序的来源可能不同。 [历史真实例子](../results/cuda-rsi-v7/agent_tools/AUDIT_PROGRESS_ZH.md)

## 5. 在已绑定用例重放导出的显存程序

使用同一发布源码及受信 `SGLangDomain` 接口。下例使用一份新身份和新回执目录的 schema 3 配置，复测训练拆分。每次复测指定唯一的 `EVAL_ID`。它会实际申请远端 GPU 租约。单次复测与正式多区块留出结论分开。

```bash
python -B - "$REPLAY_ADAPTER_CONFIG" "$EXPORTED_PROGRAM" "$EVAL_ID" <<'PY'
import json
from pathlib import Path
import sys
from rsi_memory.memory_program import MemoryProgram
from topomem_sglang_rsi.domain import SGLangDomain

config, program = (Path(value).resolve(strict=True) for value in sys.argv[1:3])
adapter = SGLangDomain(json.loads(config.read_bytes()), config.parent)
try:
    result = adapter.evaluate("train", MemoryProgram.from_dict(json.loads(program.read_bytes())), sys.argv[3])
finally:
    adapter.close()
print(json.dumps({"feasible": result["feasible"], "score_ms": result["score"],
                  "receipts": result["metrics"]["receipts"]}))
raise SystemExit(0 if result["feasible"] else 2)
PY
```

受信 `sglang_eval` 按每个用例重新编译。全驻留证明绑定实际 token IDs 摘要、每请求长度上界、固定 6144 热槽和每 rank 页预算，随后由 worker 独立重算。全驻留时裁判在各自作业目录写 `compiled_case/resident_proof.json`。候选先绑定该用例同容量的原生资格。计时再绑定同用例、同程序、同容量和同后备模式的候选资格回执。原生、手写和生成臂遵守同一 KV、调度和输出门槛。 [证明与计时边界](SG_RESIDENT_V2_ZH.md)
