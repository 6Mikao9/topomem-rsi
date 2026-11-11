# 离线拓扑 RSI 实验报告

```powershell
python -B -m rsi_memory.topology_report `
  --run "TP2=path/to/topology-rsi-run" `
  --receipts path/to/local-receipts `
  --topology path/to/topology.json `
  --scenario examples/workloads/agent-water-inspection.json `
  --scenario examples/workloads/long-document-estuary-qa.json `
  --scenario examples/workloads/chatbot-trip-planning.json `
  --scenario examples/workloads/rollout-bridge-options.json `
  --output path/to/new-report.html
```

运行、回执、拓扑和场景均可分别传多次。`--receipts` 指向本地执行器的回执目录。也可用 `--receipt` 指定个别 v2 回执。`--scenario agent_tools=path/to/memory-scenario.json` 可导入该类别的实际目标绑定及 rank 分配，界面按摘要与回执关联。直接在浏览器打开生成的 HTML。输出文件包含界面及数据，无需服务器或联网。仅提供 `--output` 可生成空状态报告。可加四个 `--scenario` 查看已定义的输入结构。

运行目录有 `journal.sqlite` 时，报告以只读方式从台账取记录，包括已结算的失败轮次与尚无结果的调用。没有 SQLite 时依次读取 `report.json` 或 JSON 检查点。生成器源码从同目录 `harnesses/` 加载，并验证字节摘要。界面分别列出真实的程序、改写源码、配对区块、封存状态、预算上限和 provider 报告的用量。主封存结果显示准入谱系，实验档案最佳另外显示验证分数与实际来源。元生成器晋级和下一轮复用在任务二中单独显示。Provider 请求和响应正文、QA 裁判答案及进程日志不嵌入。生成器归档源码本身会嵌入供核对。

评估前方案包含观察到的问题、机制改动、预期效果、证伪条件和实际选中的训练证据 ID。递归臂和固定对照均先落盘方案。固定对照的元产物弃用。预期效果属于假设，实测分数、接受或拒绝原因和版本差异另列。

GPU 观测按回执记录的后端、场景、程序和拓扑摘要显示。已导入场景文件仅表示固定输入已定义。CUDA 文本派生分块稀疏注意力、SGLang 模型服务和 CPU 参考数据各有独立的测量范围。回执需通过结果摘要核对，原始动作可使用 `--max-actions` 调整每个场景的预览条数，表中始终保留记录的动作总数。GPU 回执目录和 SQLite 台账保留完整证据。
