# SGLang 跨容量资格复核与复现

audit.json 的 artifact_sha256 提供每个原始包、源码包和独立资格 JSON 的字节摘要。raw 与 source 下的 ZIP 均为原件。只读复核可直接从 ZIP 读取结果、监护回执、两 rank 轨迹和逐层 KV 日志。

## 原环境复现

在仍匹配冻结 GPU UUID、模型字节、SGLang 安装源码、CUDA 13 与隔离 NCCL 2.32.3 的 Linux 主机上使用下列命令。SOURCE_ARCHIVE 指向本目录 source 中正式 ingress 源包。STAGING_PARENT 是新解包目录的父目录，RUN_DIR 是未存在且位于 stage 外的输出目录。PYTHON3 是宿主 Python 命令，worker Python 从封存 launch 合同取得。

```sh
set -eu
: "${SOURCE_ARCHIVE:?path to the frozen ingress source archive}"
: "${STAGING_PARENT:?directory for a fresh immutable stage}"
: "${RUN_DIR:?new run directory outside that stage}"
PYTHON3=${PYTHON3:-python3}
ARCHIVE_SHA256=110d2809fc19aa21b4d2251798833c3816cb6c9ca48c215aa085b0882f6eccc7
TRANSFER_SHA256=b63b60225e6df9e7c2d6e42ac0ccce4d67d0c0ebcfdd49cdfd3f1e7dae8ade70
printf "%s  %s\n" "$ARCHIVE_SHA256" "$SOURCE_ARCHIVE" | sha256sum --check --status
STAGE=$(mktemp -d "$STAGING_PARENT/ingress-v1-stage.XXXXXXXX")
"$PYTHON3" -m zipfile -e "$SOURCE_ARCHIVE" "$STAGE"
"$PYTHON3" -B "$STAGE/run_hot_native_parity.py" --expected-manifest-sha256 "$TRANSFER_SHA256" --verify-only
"$PYTHON3" -B "$STAGE/run_hot_native_parity.py" --expected-manifest-sha256 "$TRANSFER_SHA256" --runs "$RUN_DIR" --repeat-count 2
```

第一条 runner 命令仅校验文件。第二条才在租约下启动 GPU，按 4096、6144、8192 和反向顺序各运行一轮。完整两轮通过后输出 gated_qualification.json 和 hot_metadata_profile.json。失败时保留作业记录及失败原因。独立 metadata collector 需要连同两轮资格回执审查。

## 复核其他来源

| 源包 | 传输清单 SHA-256 | 对应原始证据 |
| --- | --- | --- |
| [topomem-sglang-hot-native-probes-v1.zip](source/topomem-sglang-hot-native-probes-v1.zip) | 574e0ff174ec88a3b9f8581037fdb423f72e351ee57e622a4369d8703ba6798d | [hot-native-parity-v1-evidence.zip](raw/hot-native-parity-v1-evidence.zip) |
| [topomem-sglang-global-tick-probes-v3.zip](source/topomem-sglang-global-tick-probes-v3.zip) | 2163d551e18a5aea21d857f22d2a2c43309b5f48fdad62434fbc3fd907add7a0 | [final-qualification-v1-ungated.zip](raw/final-qualification-v1-ungated.zip) |
| [topomem-sglang-cublas-probes-v2.zip](source/topomem-sglang-cublas-probes-v2.zip) | 7a28175678387d3bbcb6fd8633d12df15bde7dc1acd939ad99f4e008178cb662 | [final-qualification-v1-cublas.zip](raw/final-qualification-v1-cublas.zip) |

对这些归档源码也可在各自新 stage 执行 --verify-only。未加门禁的 v1 和 v3 用独立输出目录分别启动各轮。cuBLAS 试验的原件是失败回执，复核无需再改动运行环境。

封存包中的 profile-tools/sglang_launch.json 绑定原部署路径和 GPU。若模型、解释器、CUDA、NCCL、缓存或设备位置需要改变，在新部署工作区生成新的 launch、传输清单和必要的目标绑定，记录其不同摘要，再单独做资格。保留这里的原 ZIP 和清单。
