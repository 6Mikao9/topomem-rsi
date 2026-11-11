# CUDA v4.1 未完成运行

本次双卡 CUDA 工程复评在首轮训练矩阵中断。封存的三轮、每臂两槽协议使用旧公开语料和新几何。每次成功计时的单位为四个训练场景的计时中位数之和，单位毫秒。观测只涵盖训练场景。

- 首轮 10 次 API 调用已返回。出站系统提示和结构化选择合同均经过独立核查。两条元提示明确要求父代与子代分别从本轮新提案选出程序，并以历史管理器 incumbent 作独立验证参照。
- 8 个程序提案加初始锚点形成 9 个不同的声明身份。身份使用原始规范化 IR 的 SHA-256。实际布局可能相同。
- 三个预登记训练块收到 9、7、0 条观测。其中 13 条取得计时回执，其四场景选中数据与输出摘要均与本次原生训练基准一致。两个固定种子提案获得独立不可行结果。
- 子代第二槽的第二次矩阵观测遇到连接中断。控制器日志将调用记为 `ExecutionError`，连接日志记录 `Connection reset`。公开分类为 `transport_unknown`，当时本地没有完成回执。远端执行结果在本归档中未核实。
- `run` 返回 `training_matrix_incomplete`。随后 `freeze` 因轮次不完整退出。首轮没有提交验证、没有封存最终程序、没有源码晋级，也没有留出成绩。已有部分计时仅供核查。

[audit.json](audit.json) 保留出站合同证明、已验证回执摘要和流程状态。[matrix-observations.json](matrix-observations.json) 保留预登记顺序、全部提案来源摘要和已取得的块级计时。回执证据是 GPU 校验器生成的哈希。两个 Python 脚本在持有完整本地账本、冻结源码、对应配置和原始回执时可重放。本目录提供经过筛选的证据，原始账本、网络日志和完整 API 内容保存在实验本地。

```powershell
python -B audit_incomplete_v41.py --run <local-run-directory> --source <frozen-controller-source> --config <local-config> --output <new-audit-json>
```
