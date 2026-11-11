# SGLang v3 生成合同修正

v2 正式 R0 的真实模型请求同时包含通用 `allocate/place` 必选规则和 SGLang resident 模式禁止 `allocate/place` 的受信语义。R0 parent slot 0 返回热容量、后备模式和分层放置三类规则，原编译器以 `resident backing cannot include cold placement rules` 拒绝。v2 停止后保留已有台账，未结算的调用保留其当时状态。

v3 使用独立控制器入口。它通过正式 worker 使用的原 MemoryProgram parser 查明受信后端的最小合法规则形态。CUDA 原入口和必选 `allocate/place` 约束保持原字节。SGLang 请求使用互斥的 tiered 与 resident 程序形态。tiered 需要分层放置规则。resident 要求选择已资格的 6144 热槽并明确选择 resident 后备，完整程序中禁止所有分层放置规则。规则适用于组装后的程序，`mutate` 仍一次只能替换一条规则，`construct` 可以定义完整程序。

`rsi_memory/topology_rsi_contract.py` 将最终生成合同和两个新增模块的实际字节摘要写入实验 manifest，再调用原控制器。旧裁判、生成器种子、数据、派发、程序编译、paired 阈值和后备原语保持封存字节。正式 worker ZIP 和 38 份训练手写回执仍绑定原源码，构建新 schema-3 时按同一裁判重放。新的实验目录与 ID 隔离 v2 中止作业。该修正属于人工实验工程，正式生成结果与递归晋级仍需单独核验。

```text
python -B -m topomem_sglang_rsi.contract_v3 run --adapter-config <new-schema3.json> --output <new-controller-dir> --key-file <local-key>
python -B -m topomem_sglang_rsi.contract_v3 freeze --adapter-config <new-schema3.json> --output <new-controller-dir>
python -B -m topomem_sglang_rsi.contract_v3 test --adapter-config <new-schema3.json> --output <new-controller-dir>
```

离线回归从实际 controller 请求入口截取本地替身 provider 的 envelope，核对 manifest 的相同合同。原编译器接收无分层规则的 resident 程序，拒绝混合程序。回归未调用模型 API 或运行 GPU。
