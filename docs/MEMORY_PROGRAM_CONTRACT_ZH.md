# 多卡 KV 显存程序接口

`HardwareTopology`、`MemoryScenario` 和 `MemoryProgram` 分别绑定硬件资源、外部场景和生成的显存管理规则。接口表示多卡、共享 host backing 与设备间的方向性路径；每个执行后端独立声明实际支持的动作，并保存探测、部署与测量证据。[硬件拓扑](../rsi_memory/topology.py) · [程序接口](../rsi_memory/memory_program.py)

## 固定输入

- 硬件图使用 `gpu`、`host` 和 `fabric` 节点。`capacity_bytes` 表示该执行器可使用的上限。`physical_bytes` 可以记录物理总量。方向性 link 绑定 transport、带宽、延迟和一个或多个共享瓶颈。节点、链路、瓶颈可附实测证据 SHA-256。图里没有直达链路时，执行器只能考虑图中实际允许的分段路径。普通静态校验接受合成拓扑，真实运行需要 adapter 验证设备身份和路径可用性。
- 场景绑定 `target_sha256`、`workload_sha256` 和 `schedule_sha256`，提供固定的 rank→GPU、注意力类型、selector、选中粒度、Top-K 和 KV 行宽。目标哈希指向另行封存的模型与运行时契约。生成程序不修改这些字段，也不产生调度、batch 或选集决策。
- `generator_context` 返回完整拓扑、场景、执行器已声明支持的动作、特征单位与输入的实测反馈。递归生成器的代码、状态、对照和迭代门禁由其外层定义。

## 生成程序及 adapter

程序由有限个事件规则组成。`when` 和 `score` 使用有界表达式 AST。每种动作先取第一个有合法候选的规则，再取满足 `when > 0` 且分数最低的候选。稳定的候选顺序解决相同分数。容量、引用、可达性和身份由执行器判断。一个程序可以依据 rank、layer、节点占用与链路成本改变行为，例如给不同 rank 选不同层级 backing。[规则及特征](../rsi_memory/memory_program.py)

`plan_backing` 是首个真实 adapter 可消费的接口。adapter 提供经审计的 rank/layer slab、实际字节数与合法节点，结果包含逐 slab 的位置、双向链路、规则编号、每节点占用及三份绑定哈希。这个入口只接收 `allocate/place`。若程序包含当前 adapter 无法执行的动作，直接拒绝。GPU backing 的保留和选择必须在运行时验收。

## CPU 参考执行

CPU 参考对固定顺序的外部 `allocate`、`access`、`hint`、`release`、`idle` 事件执行规则，支持从在用 owner 复制、迁移、预取，从可释放副本驱逐和从退休对象回收。它用 `(rank, request, generation, layer, block)` 绑定真实的参考 payload 字节。传输期间持有源引用并预留目的与中转空间，完成后才发布副本或变更 owner。消费者验证原始有序选集及每条 KV 摘要，消费时保留引用。退休只允许回收没有读者或在途引用的副本。[CPU 执行器](../rsi_memory/memory_program_reference.py)

参考计时采用保守的链路和瓶颈独占，每条路段依次占用资源，独立路径可以重叠。它的等待时间、字节和占用只用于 CPU 机制检查，不能作为 GPU 性能数据。这个参考执行器不支持 CUDA 后端新增的压力 `demote`，遇到该动作会拒绝执行。

## CUDA 生命周期执行

[CUDA 后端](../integrations/topology_memory/torch_runtime.py) 使用真实 GPU 和 pinned-host 存储，执行固定块稀疏注意力、会话前缀延续、分支共享与写时复制。文本派生的 K/V/Q 是确定性测试数据；这里没有语言模型推理或回答质量得分。每个不可变块版本占一个管理槽，追加产生新版本；复制完整块的成本计入计时。

`allocate/demote` 与 `access/demote` 只在分配受阻时选择活跃 owner 和目的地，正在读取的选集、producer 与在途块不能作为受害者。复制并同步完成后才转移所有权并释放原位置；已有合法不可变副本可直接交接 owner。诊断运行逐字节验证交接，计时运行核对同一动作和选集签名。原生对照是本项目的 write-through/LRU 实现，另有直接执行的手写策略和同一规则后端的手写 IR。

节点容量约束包括 KV、副本、COW 和中转槽。池在测量前等额预分配，因此槽占用下降不等于向 CUDA 释放实际 HBM。公共注意力工作区单独通过真实 PyTorch 峰值报告。得分包含程序决策、数据生产、复制、注意力和强制最终回收；编译与池的预分配位于计时外。真实 SGLang 模型服务的静态层级 backing 另见 [SGLang 接入](../integrations/topology_memory/SGLANG.md)，两类结果分开报告。
