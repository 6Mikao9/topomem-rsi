# 可配置 SGLang 双卡到达组资格

这套集成在可配置 SGLang RC 上加入受信到达组门禁和 CPU 全局 forward 观察。六个原用例保持原 ID、fixture 字节、target 字节、拓扑、分词承诺和链路证据。新 case manifest 绑定门禁引入后的外部调度、场景及编译预览。正式搜索按原列表选择首个训练、首个验证和两个原留出用例。选择仅按位置，固定在首次模型调用前。其余两个用例留档。

运行前用固定本地 tokenizer 重算承诺。两 rank 在同一 ingress 投票后按原始请求对象进入 SGLang。第二组等待上一组原生状态排空，释放后额外 Gloo 投票停止。计时范围包括实测组等待。调度器、注意力、KV 管理原语和编译的存储选择保持各自的受信职责。原始、手写与生成候选都执行同一用例的门禁。

热元数据 profile 绑定共同 GPU、模型目标、运行环境、门禁实现和池几何。其校准用例的全局 forward 摘要只裁决校准回放。每个正式用例按自己的 native A/A 建立全局摘要。配对 parity 保留全部活动 tick、单 rank IDLE、输出及 656 B packed KV 门禁。正式计时绑定该用例、该臂的 parity 回执。门禁源、观察器源和编译计划均进入回执。

## 部署顺序

1. 从真实双卡三容量资格取得通过的 `hot_metadata_profile.json`。新配置拒绝旧的 ungated profile，校准和正式 worker 的门禁及全局观察实现必须相同。
2. 把 `topomem-sglang-gated-worker` 源包解压至远端实验目录内的新 source 子目录。保留 `sglang_launch.py` 可执行，私有 `sglang_launch.json` 为可信常规文件且无组或其他写权限。把六用例 ZIP 的内容解压到同一实验根下 `cases/`。依赖沿用已核验的 Python、SGLang 0.5.20、tiny GLM 模型、CUDA 及隔离 NCCL 2.32.3。worker 对这些依赖执行摘要和设备检查。
3. `make_config` 使用 `formal_case_set_manifest.json`、`formal_tokenization_receipt.json` 和正式 worker `SOURCE_MANIFEST.json`。配置中的 `--remote-source` 是实验根目录内源包子目录名。`--remote-worker` 和 `--remote-launcher` 分别指向该源包中的 `sglang_executor_worker.py` 及 `sglang_launch.py`。
4. 以同一正式源和 schema-2 配置重新执行已冻结的八策略训练手写 sweep，三个区块和 5 ms margin。先生成 preflight，再用新输出目录和其 digest 执行。保留 sweep 配置、收据、动作布局与选择记录。之前 ungated 的测量不能替代本次 sweep。
5. 用同一源码生成 schema-3 配置，传入新的 `--manual-sweep` 和 `--manual-sweep-config`。本地严格控制器复放训练回执，重新核对阈值与选择后才允许调用模型。两轮、每臂两槽、六验证区块和十二留出区块沿用原 preregistration。

```text
python3 -u <remote-source>/sglang_executor_worker.py --root <remote-experiment-root> --python <remote-source>/sglang_launch.py
```

该命令由本地客户端按原来协议启动。worker 只接受固定 SGLang evaluator 模块及合法 leased source，执行目标是 `topomem_sglang_rsi.sglang_eval`。远端依赖和私有启动配置继续与软件源码分开保管。[手写对照的原有命令及预算](SG_HAND_SWEEP.md)

## 证据范围

固定公开回放的六次双卡原生资格已通过。4096、6144 和 8192 热槽各两次，每个容量的 A/A 与跨容量核对均通过。两个 rank 的 packed KV、输出 IDs、派发及 132 个活动全局 forward tick 一致。资格 JSON 字节 SHA-256 为 `a8c385514f2ebe7732f5bffcf8e4dd4f2c66171e7a1462dea30bfec413648012`，热元数据 profile 字节 SHA-256 为 `ce1926181a763dd52882024ae27681dcd506174b1f64ebda5d006eb4a8ef9975`。[原生资格审计](../results/sglang-qualification/ingress-v1/AUDIT_ZH.md)

模型是固定的 tiny GLM 架构验证 checkpoint。服务回放为独立请求且观察到 EOS。真实训练 sweep、生成策略资格和正式 RSI 仍需各自的回执。严格 RSI 控制器、MemoryProgram 及动态 CUDA 执行器中的 `demote` 沿用封存 RC 字节。当前 SGLang 可配置域生成热容量和静态冷层放置。
