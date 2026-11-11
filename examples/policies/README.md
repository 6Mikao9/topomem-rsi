# Torch KV 策略对照

两个 JSON 都是由固定 Torch 执行器解释的 `MemoryProgram`。工件通过 `MemoryProgram.fingerprint()` 绑定，策略只接收预定拓扑和运行时已观测状态。

| 文件 | 对照用途 | 策略 |
| --- | --- | --- |
| `torch-native-lru-equivalent.json` | 相同 IR 执行路径的原生算法对照 | 权威块写入 pinned host，明确保留计算节点已写入的块，缺页时按路径代价复制，使用最近使用步驱逐。 |
| `torch-manual-topology-aware.json` | 固定手写 IR 对照 | 在保留计算节点余量时保留本地热点所有权，其他块按写路径成本和占用选 backing。优先保留生产者副本，可按上一步已观测选择预取。回收和驱逐使用近期性及频率。 |

独立运行时的 `manual_topology` 实现可作为额外对照，并采用双向路径成本与安全降级。模型生成候选和手写 IR 走同一 AST 编译、容量和传输检查。原生 LRU 等价策略在四个固定用例的 CPU Torch 参考执行中与原生算法的有序所选字节、输出和各链路字节一致。CUDA 计时和实际设备一致性需由目标机独立复核。
