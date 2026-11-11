# SGLang ingress 跨容量资格审计

正式门禁的六个原生作业全部通过。固定 DP2、模型、token IDs、两组请求与冷主存 backing，在 4,096、6,144、8,192 个热槽分别启动两次新进程。实际全局调度、选中 packed KV、输出 IDs 和运行环境摘要在同档 A/A 和跨档均完全一致。这是固定架构测试回放的正确性资格。[正式回执](gated_qualification.json) · [原始正式作业](raw/final-qualification-v1-ingress.zip)

## 原始失败链

早期未加门禁的两轮共六个作业正常退出。4,096 的 A/A 一致，6,144 和 8,192 的 A/A 均出现 packed KV 差异。原始本地派发、输出和其他比较字段仍一致。早期版本未记录全局 tick。下表只按原始结果的两 rank packed 摘要关联。

| 原始轮次 | 4,096 | 6,144 | 8,192 |
| --- | --- | --- | --- |
| 1 | A | A | B |
| 2 | A | B | A |

未加门禁的轻量 v3 另有六个作业。两 rank 的原始轨迹连续、等长，并保留单 rank 的 IDLE。B 和 C 相对 A 的首个差异在实测首个 prefill，零基索引 66，即原生 forward 67。

| 观测类 | 首个实测 prefill 的全局 token 数 | 两 rank 模式 | 每 rank 全局 tick 数 | 两 rank 的 IDLE 数 | packed KV 摘要前缀 |
| --- | --- | --- | ---: | --- | --- |
| A | [2048, 2048] | EXTEND / EXTEND | 132 | [0, 0] | 284f1a979165… / dabc2552fa6c… |
| B | [0, 2048] | IDLE / EXTEND | 133 | [1, 1] | 674148f95f5d… / 9edcae07573d… |
| C | [2048, 0] | EXTEND / IDLE | 133 | [1, 1] | a07c90823d8b… / 3d39906f8350… |

| 轻量 v3 轮次 | 4,096 | 6,144 | 8,192 |
| --- | --- | --- | --- |
| 1 | A | A | B |
| 2 | A | A | C |

关联使用原始 result 中逐 rank packed KV 摘要，并从有序逐层日志重算。哈希只排除 miss_rows，选行顺序、历史字节摘要和 newest 字节摘要完整保留。原始 KV 文件仍含缺页计数，容量不同可改变文件摘要。此组数据定位了全局批次形状和 packed KV 漂移，无法归因到单一内核。[轻量 v3 原始作业](raw/final-qualification-v1-ungated.zip) · [早期第一轮](raw/hot-native-parity-v1-evidence.zip) · [早期 A/A](raw/hot-native-parity-v1-aa-repeat-evidence.zip)

另一次诊断请求 CUBLAS_WORKSPACE_CONFIG=:16:8。首个 4,096 作业在 owned-worker 环境继承断言失败，被终止且完成进程组清理。未写出 result，未记录该 worker 当时的实际值。轻量 v3 的十二个 rank 启动快照在各自采样时均记录该变量为空。[环境继承失败](raw/final-qualification-v1-cublas.zip)

## 正式门禁与实测几何

每个正式作业重新核对两个 rank 的同票次 ingress 释放、实际 token 承诺和原生完整轨迹。六次都为每 rank 132 个活动全局 tick，实测首个 prefill 为 A，未观测到 IDLE。检查器保留 IDLE 字段并比较整个轨迹。每 rank 每作业 756 个 KV 诊断事件，覆盖两请求各 63 个 decode forward 的六层。六次 packed 结果均与观测类 A 相同。

| 热槽 | 每 rank 热元数据 B | 每 rank packed device pool B | 每 rank 逻辑设备行 |
| ---: | ---: | ---: | ---: |
| 4096 | 588,800 | 64,739,328 | 16,384 |
| 6144 | 879,616 | 64,739,328 | 16,384 |
| 8192 | 1,170,432 | 64,739,328 | 16,384 |

profile 中另记每 rank 16,448 个物理行，以及含 pad 的两个请求元数据槽。档间改变的是热保留上限和元数据，整个设备 payload 池大小相同。逐项核对 profile 绑定的首轮原始结果摘要，并复核第二轮同档实际几何。18 个完成作业的 36 个 owned-worker 回执均在真实 decode 后采到单一 NCCL 2.32.3 映射，映射的设备、inode、库 SHA 和版本一致。各完成作业返回码为 0，监护回执无进程组或 compute PID 残留。环境断言失败的作业单列，原始日志照存。[实测 profile](hot_metadata_profile.json) · [逐作业核查](audit.json)

实验使用固定微型 GLM 架构测试模型，输出为 EOS token 重复。此处未评估 SGLang 策略性能或模型任务质量。[复核与复现步骤](REPRODUCE_ZH.md)
