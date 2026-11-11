# Resident v2 原生校准复核与复现

保留本目录中的原始源码 ZIP、原始作业 ZIP 和独立资格/profile 文件。audit.json 的 artifact_sha256 列出它们的完整字节摘要，worker_manifest_files_sha256 列出同一份 27 文件清单。打开原始作业 ZIP 可按轮次与热槽复核两个 rank 的全局 tick 和 KV 诊断。

在仍满足封存模型、GPU、SGLang、CUDA 与 NCCL 条件的 Linux 主机上运行下列命令。SOURCE_ARCHIVE 指向本目录的 resident-v2-native-calibration.zip，STAGING_PARENT 是新 stage 的父目录，RUN_DIR 是未存在且位于 stage 外的输出目录。宿主解释器可设 PYTHON3，worker 解释器从绑定 launch 文件读取。

```sh
set -eu
: "${SOURCE_ARCHIVE:?path to resident-v2-native-calibration.zip}"
: "${STAGING_PARENT:?parent of a new immutable stage}"
: "${RUN_DIR:?new output directory outside the stage}"
PYTHON3=${PYTHON3:-python3}
ARCHIVE_SHA256=065ff01df373d5ded7a3d1a2a185afae81b6c16af9d912486f613c97ab68e066
TRANSFER_SHA256=bbfffcf215e5bf9da4dc2e6c2b4ca45a55a90bb06bfc3cad71f3d1596229f43e
printf "%s  %s\n" "$ARCHIVE_SHA256" "$SOURCE_ARCHIVE" | sha256sum --check --status
STAGE=$(mktemp -d "$STAGING_PARENT/ingress-resident-v2.XXXXXXXX")
"$PYTHON3" -m zipfile -e "$SOURCE_ARCHIVE" "$STAGE"
"$PYTHON3" -B "$STAGE/run_hot_native_parity.py" --expected-manifest-sha256 "$TRANSFER_SHA256" --verify-only
"$PYTHON3" -B "$STAGE/run_hot_native_parity.py" --expected-manifest-sha256 "$TRANSFER_SHA256" --runs "$RUN_DIR" --repeat-count 2
```

verify-only 只核对源码和 case，第二条 runner 命令才持有 GPU 租约并启动六个原生进程。完整两轮通过后输出 gated_qualification.json 和 hot_metadata_profile.json。复核 profile 时一并保留完整两轮资格回执。

原 launch 文件绑定模型位置、CUDA、NCCL、缓存和 GPU。若部署条件变化，在独立 stage 重新绑定对应文件与摘要，并在该源码上重新取得原生资格。保留本目录的原始包和原清单。
