# 服务器逐作业证据

本组有 226 个作业、6970 个原成员。各 ZIP 原字节与收回时一致。归档逐成员重哈希并与本地租约回执对上请求、进程日志、evaluator 与 serving 原结果。复算 226 个全局转发及 ingress 记录，并从日志复算 40 个资格作业的双 rank packed KV 摘要。

在仓库根目录执行复核。

```sh
python -B results/sglang-rsi-v3/tools/verify_sglang_final_server_evidence.py --server-export results/sglang-rsi-v3/server-evidence --source-archive results/sglang-rsi-v3/frozen-source.zip --destination results/sglang-rsi-v3 --group formal-v3 --verify-only
```

[本组逐成员和作业审计](index.json) · [原服务器归档索引](evidence-archive-index.json) · [发布工件摘要](../artifact_manifest.json)
