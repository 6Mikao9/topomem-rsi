# 手写策略选优复核与重放

[工件清单](artifact_manifest.json) 给出本目录原始文件、ZIP 和 38 个成员的字节 SHA-256。[公开 adapter contract](adapter_contract.json) 是冻结控制器实际用于重建作业 ID 的结构，规范 JSON SHA-256 为 `0f0da5b5221d1cc907bac672c55f7683b4434e3794e3881c55672049a3094dec`。[schema 2 模板](schema2.template.json) 保留训练、验证、测试 case 的承诺及编译 profile。operator 位置与 GPU UUID 以占位符表示。原私有配置的字节 SHA-256 为 `df8c7830f0cc9794e153fc3ecfe27a1c01264294e445bcc818ab0a9bf2cb9fe1`，原 launch 的字节 SHA-256 为 `ad719ed44db6b8722688cc17bc74ea0b0555f4c8590e9bb0854afe142b9c63ab`。两个私有文件均由部署方持有。重新填写模板会改变字节摘要，异机绑定也可能改变作业 ID、运行环境和资格。

## 离线检查

在本目录运行以下标准库脚本可校验保存的字节、ZIP 成员及 27 个原始窗口的选优结论。审计还用冻结控制器和只读回执替身重放了 38 个请求，源文件 127 个及 worker 27 个均对照封存清单重哈希。重建完整历史请求摘要另需原私有配置里的 remote source label 和 UUID 绑定。

```python
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

manifest = json.loads(Path("artifact_manifest.json").read_bytes())
for name, expected in manifest["files_sha256"].items():
    assert sha256(Path(name).read_bytes()).hexdigest() == expected, name
with ZipFile("leased-receipts.zip") as archive:
    assert set(archive.namelist()) == set(manifest["leased_receipt_members_sha256"])
    for name, expected in manifest["leased_receipt_members_sha256"].items():
        assert sha256(archive.read(name)).hexdigest() == expected, name

sweep = json.loads(Path("sweep/manifest.json").read_bytes())
result = json.loads(Path("sweep/result.json").read_bytes())
jobs = [json.loads(line) for line in Path("sweep/jobs.jsonl").read_bytes().splitlines()]
windows = {label: [] for label in sweep["declared_policy_order"]}
for block, order in enumerate(sweep["order"], 1):
    timed = [job for job in jobs if job.get("block") == block]
    assert [job["label"] for job in timed] == order
    for job in timed:
        windows[job["label"]].append(job["window_ms"])
assert windows == result["sum_window_ms_by_block"]
reference = windows["stock-6144-host"]
gains = {label: [base - val for base, val in zip(reference, values)]
         for label, values in windows.items()}
eligible = [label for label in gains if min(gains[label]) > 0 and median(gains[label]) >= 5]
assert min(eligible, key=lambda label: (-median(gains[label]), label)) == result["selected"]
default = windows["stock-4096-host"]
assert max(5, 2 * (max(default) - min(default))) == 11.49229000000014
```

## 参数化实机重放

1. 解开 [本地控制器源码 ZIP](source/resident-v2-rc.source.zip)，核对其中 `LOCAL_SOURCE_MANIFEST.json` 与 [单独原件](source/resident-v2-rc.source-manifest.json) 一致。远端使用 [已归档的 worker ZIP](../sglang-qualification/ingress-resident-v2/source/resident-v2-worker.zip)，其 `SOURCE_MANIFEST.json` SHA-256 为 `b73995aea418432831366a7c2de0af038fbe8eae2f2f85ac164d2e88e4c81909`。解到新远端实验根的一个子目录，将子目录名填入 `executor.source`。将该源码中的 `sglang_executor_worker.py` 路径填入 `executor.worker`，`sglang_launch.py` 的可执行路径填入 `executor.python`。入口相邻的可信常规文件 `sglang_launch.json` 使用 schema 2，字段为 `schema, python, model, nccl_isolate, cuda_home, cache_root, gpu_indices, gpu_uuids`。其中 `python` 指向实际 SGLang 虚拟环境解释器，`gpu_indices` 为 `[0, 1]`。用相同模型内容与 NCCL 隔离清单，核对 worker 所见 UUID 顺序及动态库字节。
2. 部署方将冻结 case transfer 包解到新远端根的 `cases/`，并保留一份同一套 case 给 `local_cases`。原包由部署方持有，其 SHA-256 为 `921f665566bd6b3e1223954b1a0b5c2ef43711b47ef46dbab836c8f5ca6a4e65`，case 与 token 承诺见公开模板。模板内嵌 [原生校准 profile](../sglang-qualification/ingress-resident-v2/hot_metadata_profile.json) 的原值。使用新设备、runtime/source 或更换 case 时，重新冻结目标及 tokenization 绑定，并按 [原生校准流程](../sglang-qualification/ingress-resident-v2/REPRODUCE_ZH.md) 资格化。
3. 复制 `schema2.template.json` 到部署方私有目录，补齐占位字段，令 `launch_sha256` 等于实际 launch 文件的字节 SHA-256，并准备全新的本地回执目录与远端实验根。先生成和审查 preflight。以下命令在解开的本地控制器根目录运行。

```sh
set -eu
: "${ADAPTER_CONFIG:?path to the deployment-specific schema2.json}"
: "${PREFLIGHT_DIR:?new preflight output directory}"
: "${SWEEP_DIR:?new timed sweep output directory}"
PYTHON3=${PYTHON3:-python3}
"$PYTHON3" -B -m topomem_sglang_rsi.manual_sweep --adapter-config "$ADAPTER_CONFIG" \
  --output "$PREFLIGHT_DIR" --sweep-id gated-resident-hand-control-v2 --blocks 3 --minimum-gain-ms 5
```

原运行 preflight 的规范 `sweep_sha256` 为 `261e75114485091e943680b0c7d5e48eb65fbb31366bbd18d00de604273bec89`。审查本次清单后，用本次输出的规范摘要替换 `REVIEWED_PREFLIGHT_SHA256`。`--execute` 只执行训练 case，预登记本例上限 38 次 GPU 作业。

```sh
: "${REVIEWED_PREFLIGHT_SHA256:?canonical digest printed by the reviewed preflight}"
"$PYTHON3" -B -m topomem_sglang_rsi.manual_sweep --adapter-config "$ADAPTER_CONFIG" \
  --output "$SWEEP_DIR" --sweep-id gated-resident-hand-control-v2 --blocks 3 --minimum-gain-ms 5 \
  --execute "$REVIEWED_PREFLIGHT_SHA256"
```
