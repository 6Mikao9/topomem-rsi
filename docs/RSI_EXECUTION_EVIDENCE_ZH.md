# RSI 执行与评价的上游源码核对

## SoL-Pi 的实验和发布边界

研究中，外层将假设分配到独立机制谱系，内层执行提出、实现、审查、运行、分析和保留或舍弃的循环。研究从 152 个方向起步，最后有四个机制进入 harness。训练环境包括 495 个由 issue 与 PR 构造的回归任务和 40 个基于可执行裁判生成的任务。候选和接受规则封存后才执行保留集，测试反馈不回到搜索。论文叙述的研究流程本身固定，单次实例可修改的临时编排仅在该轮存活。这些结果支持机制发现和独立验证，对生成器跨轮更新的收益仍需另做实验。[SoL-Pi 论文](https://arxiv.org/abs/2609.20519) · [项目实验方法](https://nvlabs.github.io/SoL-Pi/#method)

公开仓库 `219568034fa54ed121ede80961655e33dfb9d0e3` 的 `registerConfiguredFeatures` 注册四个 Pi 扩展。`validateReceipt` 将模型压缩后的证据与归档 SHA、观测到的退出状态和原始日志引文核对，并拒绝隐去已知失败的摘要。`decideCompaction` 估计预期请求数、压缩节省、写缓存开销和偿还周期。公开集成测试以真实 Pi 会话配合确定性测试 provider 核验兼容性。此次审核未在公开树中找到论文外层的实验编排、隔离裁判或真实 GPU 评测器。[扩展入口](https://github.com/NVlabs/SoL-Pi/blob/219568034fa54ed121ede80961655e33dfb9d0e3/src/sol-pi/index.ts) · [证据核验](https://github.com/NVlabs/SoL-Pi/blob/219568034fa54ed121ede80961655e33dfb9d0e3/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts) · [压缩经济性](https://github.com/NVlabs/SoL-Pi/blob/219568034fa54ed121ede80961655e33dfb9d0e3/src/sol-pi/extensions/online-context-compact/economics.ts) · [集成测试范围](https://github.com/NVlabs/SoL-Pi/blob/219568034fa54ed121ede80961655e33dfb9d0e3/docs/compatibility.md)

群体优化示例使用 Anthropic 的机器模拟器周期数。相同冻结起点下，每个配置只运行一次两小时。SoL-Pi 工作组为 1,127 周期、$60.11，Pi 工作组为 1,366 周期、$82.12，单智能体为 1,333 周期、$39.20。两个工作组共享同样的协调者和工作者设置，单智能体总成本更低。此例能支持使用独立检查决定晋级和保留多个探索谱系。GPU 稳定收益和跨轮生成器进步均须另测。[SoL-Pi 群体实验](https://nvlabs.github.io/SoL-Pi/#swarm)

## 显存任务可复用的分层机制

- **第一个任务产出策略。** 固定目标硬件、attention、实际执行路径与外部场景。接受前由裁判复核输出、选中 KV、所有权、容量和流依赖，再测同等原生条件下的收益。NVIDIA 的独立项目 SOL-ExecBench 先执行正确性和防篡改检查，再运行计时，可另测参考实现。其 `time_runnable` 使用不同输入地址，并默认以 CUPTI 定位 kernel、拷贝和清零的活动序列。这些适合作为隔离的微基准机制，服务层应另外按预定冷热条件测端到端时间、各 GPU 峰值显存和传输量。[评测入口](https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/driver/templates/eval_driver.py) · [GPU 计时实现](https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/timing.py)
- **第二个任务改进产出策略的过程。** SoL-Pi 的独立谱系和失败日志证据能帮助生成器保留有归因的探索结果。RRSI 的 `schedule.edit_budget` 逐步降低每轮独立修改上限，以便后期归因。Dream-RSI 的 `learning/discover.py` 让 agent 改写产出程序，由回放比较旧策略和候选，再在后续在线轮执行晋级策略。迁移到本项目时，保留每轮生成器差异和产出差异、完整失败记录、预算匹配的固定生成器对照以及隔离的评估。回放晋级和实机胜出分开记录。[RRSI 编辑预算](https://github.com/google-research/rrsi/blob/main/rrsi/schedule.py) · [Dream-RSI 离线策略发现](https://github.com/google-research/dream-rsi/blob/main/dream_rsi/learning/discover.py)
