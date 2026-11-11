# 固定外部场景输入

四份输入分别定义工具代理、长文问答、对话和分支 rollout。文本和工具响应均为本仓库编写的 CC0-1.0 参考数据。长文样本借鉴长上下文阅读问答的任务形式，样本文本、问题和答案均独立编写。[LongBench 项目](https://github.com/THUDM/LongBench)

| 输入 | CPU 参考验证 | SGLang 完整场景资格所需的运行时证据 |
| --- | --- | --- |
| `agent-water-inspection.json` | 两个 ID 匹配的工具调用和返回，三次推理之间保留同一会话前缀的页引用 | 外部驱动真实经过工具接口或固定响应回放，记录每次调用和返回 ID、追加 token、已复用 KV 行及其内容。 |
| `long-document-estuary-qa.json` | 长文、独立问题及裁判保管的参考答案，检查从长文选出的有序字节 | 固定 tokenizer 和模板，记录完整文档的 token 数、截断状态、模型答案及预注册评分，同时核验实际注意力选择的 KV。 |
| `chatbot-trip-planning.json` | 三轮增长，同一会话前缀跨轮保留 | 固定会话和轮次，记录每轮消息、token 长度、保留前缀命中与实际复用的 KV 行。 |
| `rollout-bridge-options.json` | 两个子分支共享父页，在未填满的末页首次写入时分别复制，前缀字节不变且分支后缀不同 | 在同一运行时记录 parent/child、共享页引用数、写前复制、独立写入后的各分支有序 KV 字节，以及全部引用释放。 |

`ScenarioWorkload.read()` 校验内容和事件。`workload_sha256` 绑定全部场景字段，`schedule_sha256` 绑定按序事件、相对时间和每次输出预算。实际驱动应封存 tokenizer、模板、外部调度参数和 token 化后的输入。QA 的 `task.expected_answers` 由裁判持有，生成器和模型输入不读取该字段。`select_rows` 是 CPU 参考中的有序 UTF-8 字节位置，生产注意力选择以目标自身的固定 selector 与运行时日志为准。参考中的 `page_rows` 也仅描述 UTF-8 参考页。

`.reference_report()` 的执行范围为 `cpu_utf8_page_reference`。它逐次比较物理页存储和不可变逻辑文本的字节，记录各场景的见证事件、前缀与所选行哈希，并检查最后归零的页所有权。`.require_backend_capabilities(...)` 只检查后端声明支持哪些语义。实际运行还需从独立 GPU 证据核对上表，并核验固定外部输入下原生与生成臂的输出、引用安全和容量上限。当前 SGLang 独立请求和按层 backing 实验未登记以上四项完整场景资格。独立重提交流程可单独保留为阶段性对照。

## 独立训练和保留集

`rsi_memory.experiment_cases` 分别构建四个 RSI 域。各域的训练集使用上表原始内容和两个封存 K/V 种子。验证集和测试集各含两份独立编写的内容，工具回合数和回包长度、对话轮数、文档和干扰事实、分支数及尾页行为有所变化。种子从实验名、场景、分组、用例序号和工作负载哈希派生。训练种子重复检验同一内容下的不同张量，验证和测试检验新内容及生命周期。验证内容和测试内容留在裁判侧。

配置构建需要实测 pinned-host 双向复制报告、每个场景的节点预算和固定计算位置、运行设备映射，以及预先固定的原生等价与手写 IR 工件。构建器回读报告原文件，核对链路速率、延迟、设备物理容量和报告 SHA。当前报告只测逐向复制，配置没有登记未测量的共享拥塞或直连链路。输出配置还包含执行器的私有连接字段，应保存在实验控制端。

```powershell
python -B -m rsi_memory.experiment_cases --inputs measured-inputs.json --output new-private-configs
```

`measured-inputs.json` 的字段与 `build_configs` 关键字参数一致。`profiles` 按四种场景分别提供 `topology`、`target`、`compute_node`、`device_map` 和 `compile`。配置构建只校验输入并生成文件。下列测试单独运行 CPU 参考。

```powershell
python -B -m unittest discover -s tests -p test_scenarios.py -v
```
