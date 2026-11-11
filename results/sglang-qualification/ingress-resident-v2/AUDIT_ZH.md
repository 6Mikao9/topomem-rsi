# Resident v2 源码的原生跨容量资格

同一版 worker 在 4,096、6,144、8,192 热槽各运行两次。六个作业均为 stock HiSparse 原生全主存 backing，通过同档 A/A 和跨档精确核对。此处资格用于绑定新版源码的原生校准。未运行生成策略或 resident backing，未评估服务性能。[资格回执](gated_qualification.json) · [原始作业](raw/resident-native-parity-v2-evidence.zip)

## 原始回执核对

- 第一轮实际顺序为 4096、6144、8192，第二轮反向。重算六个 result 文件摘要并与监护回执、资格汇总和 profile 对齐。12 个独立 rank 进程均在真实 decode 后记录单一 NCCL 2.32.3 映射。逐条检查映射设备、inode 和库字节摘要。六作业均返回 0，监护回执无进程组或 compute PID 残留。
- 逐 rank 重算外部调度、实际派发、token 承诺、输出 ID、全局轨迹和 packed KV 摘要。每次每 rank 132 个活动 tick，实测首个 prefill 为零基索引 66，双 rank token 数为 [2048, 2048]。两 rank 逐 tick 对齐，六份完整轨迹一致。原始 IDLE 字段保留，此次没有 IDLE。每 rank 每作业 756 个 KV 事件。重算摘要时只去掉 miss_rows，保留选行顺序、历史和 newest 字节摘要。
- 三套 27 文件清单和各成员字节一致，分别来自校准包 hot-control-src、profile-tools 和正式 worker 包。六个原始 runtime 均报告两个新版 resident helper 的源码摘要。实际回执均为 tiered、无 resident binding，每 rank pinned host payload 为 129,226,752 B。源码记录和实际 backing 分别验证。[逐作业审计](audit.json) · [校准源码](source/resident-v2-native-calibration.zip) · [worker 源码](source/resident-v2-worker.zip)

## Profile 与版本边界

| 热槽 | 每 rank 热元数据 B | 每 rank packed device pool B | 每 rank 物理／逻辑行 |
| ---: | ---: | ---: | ---: |
| 4096 | 588,800 | 64,739,328 | 16,448／16,384 |
| 6144 | 879,616 | 64,739,328 | 16,448／16,384 |
| 8192 | 1,170,432 | 64,739,328 | 16,448／16,384 |

两轮实测几何一致，profile 每档绑定首轮原始结果与该档 case 清单。新版校准用例与 ingress v1 封存文件逐字节相同，选中 KV 和实际全局轨迹也一致。新版运行记录增加了实际 worker limits，effective scheduler 摘要随记录结构改变，其余共有 scheduler 字段相同。新版 profile 绑定本次的完整 runtime 和源码集合。离线用封存 worker 验证器确认本 profile 可用，旧 ce192618… profile 因源码绑定不符被拒绝。[新版 profile](hot_metadata_profile.json) · [旧版审计](../ingress-v1/AUDIT_ZH.md)

数据适用固定微型 GLM 架构测试回放。该回放固定生成 EOS token，正确性资格以实际选行、packed 字节和全局调度为边界。[参数化复核与复现](REPRODUCE_ZH.md)
