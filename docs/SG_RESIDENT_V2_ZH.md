# SGLang 双卡 resident v2 部署与资格

这套独立源码沿用双 rank ingress rendezvous、固定 DP2 全局 forward 观测、严格 RSI 控制器及原分词承诺。六个原始用例和正式四用例的身份、外部调度、拓扑、输入与模型保持原字节。正式集合按原顺序取首个训练、首个验证和两个留出。v2 需要自己的原生 profile、手写对照和正式回执。

## 新增受信原语

`configure/backing_mode` 用一个有界标量规则从 `tiered` 和 `resident` 选择。默认 `tiered`，执行既有静态冷 KV 层放置。选择 resident 要求另选已资格的 6144 热槽，程序不得含 `allocate/place`，所有固定请求的实际分词承诺加生成上限均须能进入热槽。每 rank 对预热与测量请求分别按 64 行页向上取整，触及热槽上限时加原生额外页，再合计检查原生物理分配器容量。控制器仅向生成器提供训练用例特征。工作进程使用实际 IDs 独立重算证明。

resident 沿用原生设备热 KV 和 DSA attention，跳过已证明多余的冷 KV 分配与备份。每 rank 的六个独立 CUDA 指针占位逻辑合计 3936 B。账本按每个小分配可能需要独立 2 MiB CUDA allocator segment 保守计 12 MiB，另计实测热元数据相对默认容量的增量。回执分别记录 CUDA 实际 allocated 与 reserved 增量。编译产物内的 all-host 冷布局在此模式只用于原生几何与 parity 对照，实际冷 payload 为零。有效程序指纹包含存储模式。

parity 阶段依据模型 KV 生产者观察得到独立 packed bytes 与实际物理地址，核对每次真实 decode 的选择及 newest。把验证过的 656 B 行按既有原生事件格式摘要，再比较双 rank 的全局 forward、派发和输出。计时阶段绑定同源同臂 parity 回执并关闭 producer oracle。候选和固定对照均由同一受信裁判处理。

已归档的 resident v3 工程探针使用公开固定回放，四次 parity 与四次计时全部通过。预先封存的工程门槛要求 stock/resident、resident/stock 两对分别快过 5 ms，实测增益为 28.002916 与 28.479115 ms。这仅决定将原语加入 v2 候选和手写集合。正式新源码的全部资格、搜索及配对留出需独立完成。checkpoint 为 tiny GLM 架构测试模型，回放输出为 EOS。

## 同源原生资格

使用私有 launcher 从本目录构建新的正式 worker ZIP。另用 `pack_resident_inputs.py` 将该 worker 的逐字节相同副本装入校准包的 `hot-control-src` 与 `profile-tools`，并生成新的 formal cases 转移清单。校准复用先前固定三容量病例和原生 runner，执行顺序为 4096/6144/8192 与 8192/6144/4096 两轮。所有比较通过才生成 profile。profile 绑定完整 runtime identity 和实际 11 个 serving adapter 文件摘要。

```text
python3 -B run_hot_native_parity.py --expected-manifest-sha256 <calibration-transfer-manifest> --verify-only
python3 -B run_hot_native_parity.py --expected-manifest-sha256 <calibration-transfer-manifest> --runs <fresh-output-outside-stage> --repeat-count 2
```

正式 worker 与 `cases/` 分别置于新的专用实验根。保留私有 launcher 的模型、CUDA 和隔离 NCCL 路径与权限。正式 config 使用新 worker 的 `SOURCE_MANIFEST.json`、刚测的 profile 和新 cases 转移包中的原 formal case set 及 token receipt。原 v1 源、旧 profile 和历史回执分别保留。

## 九策略训练对照与正式协议

声明顺序沿用八个手写策略，并在末尾加入 resident 6144。计时顺序从重新生成的九臂确定性序列取三个预登记区块，完整均衡需 18 个区块。每个区块包含两个原生容量、六个各在每 rank 本地存一层的冷布局和一个 resident 6144。可行臂先通过与相同容量 native 的 parity，跨容量 native 继续进行 A/A 与逐字段检查。一个训练用例且全部可行时最多 38 个独立 GPU 作业。相对 native 6144 stock，候选须每个区块均有正收益且配对中位数至少 5 ms 才可替换手写参考。

保留 schema 2、preflight canonical SHA、sweep 输出及 leased receipts，再用 `make_config --manual-sweep ... --manual-sweep-config ...` 生成 schema 3。裁判离线重建九策略训练结果和选择。正式协议为两轮、每臂两槽、六个配对验证区块、十二个封存后留出区块。最低改善为 5 ms 和三份默认 native 训练分值极差两倍中的较大者。

九策略的 38 个训练作业全部结算，选中 `resident-6144`，得到 11.492290 ms 的正式门槛。正式 v2 搜索在发现通用提示的 `allocate/place` 必选约束与 resident 无冷放置的条件冲突后停止。该尝试只保留原生基线和部分生成尝试，候选配对与留出均未开启。修订后的正式控制器需要重新封存。 [训练选优审计](../results/sglang-hand-sweep-v2/README_ZH.md) · [复现流程](../results/sglang-hand-sweep-v2/REPRODUCE_ZH.md)
