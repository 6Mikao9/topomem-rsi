# 服务器逐作业证据

本组有 42 个作业、1270 个原成员。各 ZIP 原字节与收回时一致。归档逐成员重哈希并与本地租约回执对上请求、进程日志、evaluator 与 serving 原结果。复算 42 个全局转发及 ingress 记录，并从日志复算 10 个资格作业的双 rank packed KV 摘要。

在仓库根目录执行复核。

```sh
python -B results/sglang-native6144-ablation-v1/tools/verify_sglang_final_server_evidence.py --server-export results/sglang-native6144-ablation-v1/server-evidence --source-archive results/sglang-rsi-v3/frozen-source.zip --destination results/sglang-native6144-ablation-v1 --group native6144-supplement-v1 --verify-only
```

[本组逐成员和作业审计](index.json) · [原服务器归档索引](evidence-archive-index.json) · [发布工件摘要](../artifact_manifest.json)
