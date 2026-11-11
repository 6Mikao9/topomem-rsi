# CUDA v4.1 双卡工程复评中断

四个已公开场景采用 H256。每案在预置的单张计算卡执行注意力，四案计算设备依次为 GPU0、GPU1、GPU0、GPU1。独立实验共计划三轮，每臂两槽。第一轮在训练矩阵中遇到连接重置，保留 9、7、0 条区块观测。10 次模型调用均返回。流程记录 `run` 返回 0 并报告矩阵不完整，`freeze` 返回 1。原生验证基准已有回执。候选验证、提交轮次、来源晋级、程序冻结及留出执行均未发生。

本次保存 26 份本地原始租约回执，其中 15 份属于部分训练矩阵。另有四份启动资格回执来自 [此前提示中断试验](../prompt-v4-abort/README_ZH.md)，均为复用。训练回执只支持逐次观测，不形成完整配对比较。冻结源码用封存的外部效果离线重放，重现原报告、账本和 `freeze` 拒绝。

未知作业的标识由原效果输入与请求合同重算，与恢复检查完全一致。宿主机重启后、恢复容器服务前复制了其目录。四个原文件均为 0 字节。该作业没有本地回执，也没有远端完成结果。保留 `transport_unknown`，没有补测或拼接矩阵。

[原报告](controller/report.json) · [流程状态](pipeline-status.json) · [复核结果](audit.json) · [独立矩阵审计](independent/v41-incomplete-public/README.md) · [恢复检查](recovery/recovery-inspection.json) · [原租约回执](leased-receipts.zip) · [公开配置模板](inputs/adapter-config.template.json) · [工件清单](artifact_manifest.json)

原私有配置只保留 SHA-256。完整源码重放需要原配置；公开配置模板供新实验填写本机连接。公开文件与归档成员可离线核验。

```sh
python -B tools/export_topomem_v41_reboot_interruption.py verify --publication .
```
