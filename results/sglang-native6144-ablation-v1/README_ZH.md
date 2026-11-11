# SGLang 冻结后补充对照

使用正式测试已经使用的两个 case，对照冻结的 adaptive 程序与原生 6144 全 host。这个补充测量保持已选程序，证据范围是重复使用的测试 case。

运行状态 **completed**。已核对 42 份本地租约回执。

等待器终态 SHA-256 为 bb94329005527763ab85e52d63654b8331ea80938db8bb83b2a24d4a03c125b5。

8 个完整区块中有 5 个区块窗口缩短，中位差为 11.923439 ms。预登记门槛判定为 **未通过**。两臂至少一个 case 的实际热容量不同。实际 backing 与热容量见原结果。

[预注册](preregistration.json) · [等待器终态](waiter/completion.json) · [原租约回执 ZIP](leased-receipts.zip) · [工件清单](artifact_manifest.json)

[补充运行原始结果](result.json)

本地查看器用正式公开目录中的 controller 和本目录一起渲染。

```sh
python -m rsi_memory.topology_report --run FORMAL_EXPORT/controller --ablation THIS_EXPORT --output NEW_REPORT.html
```

服务器原件包含 42 个作业，其中 32 个为本试验新增计时，10 个资格作业来自同源正式运行。逐案核对显示两案 adaptive 均为 4096 slots，原生对照为 6144 slots。[服务器证据](server-evidence/README_ZH.md) · [正式留出结果](../sglang-rsi-v3/README_ZH.md)
