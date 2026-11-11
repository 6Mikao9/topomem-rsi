# CUDA 后续作业服务器证据

五组归档覆盖 219 份本地已完成租约。已验证 27 个原始 ZIP 和 1,314 个数据成员的字节数与 SHA-256。各组索引绑定原结果目录的工件清单。

| 归档 | 作业 | 服务器完整匹配 | ZIP |
|---|---:|---:|---:|
| [v12](v12/README.md) | 24 | 24 | 3 |
| [v13](v13/README.md) | 24 | 24 | 3 |
| [prompt-v4-abort](prompt-v4-abort/README.md) | 11 | 11 | 2 |
| [v41-reboot-interruption](v41-reboot-interruption/README.md) | 26 | 25 | 3 |
| [final-gpu3](final-gpu3/README.md) | 134 | 134 | 16 |

双卡重启组的 `eval-960140543890d7588593c2bc332367896d8a1183b41a540e` 有已完成的本地回执。服务器副本的 `result.json` 和 `receipt.json` 均为 0 字节，原样保留。服务器的 input、request、started 和 process log 均与本地租约吻合。此前单列的 `transport_unknown` 作业仍在[双卡中断恢复归档](../../cuda-v41-reboot-interruption/README_ZH.md)中。

[总索引](index.json)记录各组审计摘要。各组 README 提供无需 GPU 或服务器连接的复核命令。
