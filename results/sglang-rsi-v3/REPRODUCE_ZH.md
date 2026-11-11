# 复核正式与补充结果

从仓库根目录执行以下命令。摘要清单校验归档的原始字节，materialize 选项从已验证 ZIP 释放本地 viewer 需要的回执。已有回执必须与 ZIP 相同。

```sh
python -B results/sglang-rsi-v3/tools/verify_sglang_publication.py results/sglang-rsi-v3 --materialize-receipts
python -B results/sglang-native6144-ablation-v1/tools/verify_sglang_publication.py results/sglang-native6144-ablation-v1 --materialize-receipts
python -B -m rsi_memory.topology_report --run results/sglang-rsi-v3/controller --receipts results/sglang-rsi-v3/receipts --ablation results/sglang-native6144-ablation-v1 --output sglang-results.html
```

历史 schema 3 由操作方持有，发布模板保留其内容绑定及手写策略选择证明。使用归档四案作为固定请求来源，执行入口来自冻结源的 topomem_sglang_rsi.contract_v3。验证与留出块 ID、模型预算和实用门槛从 schema 3 的预登记字段读取。新部署使用自己的可信连接与启动文件，并按重新绑定后的输入与 GPU 配置资格化。

```sh
python -B -m topomem_sglang_rsi.contract_v3 run --adapter-config "$ADAPTER_CONFIG" --output "$NEW_CONTROLLER_DIR" --key-file "$OPERATOR_KEY_FILE"
python -B -m topomem_sglang_rsi.contract_v3 freeze --adapter-config "$ADAPTER_CONFIG" --output "$NEW_CONTROLLER_DIR"
python -B -m topomem_sglang_rsi.contract_v3 test --adapter-config "$ADAPTER_CONFIG" --output "$NEW_CONTROLLER_DIR"
```

[四案部署与分词说明](formal-cases/README_ZH.md) · [源和输入摘要](artifact_manifest.json) · [服务器原事件核对](server-evidence/README_ZH.md)
