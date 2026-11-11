# SGLang 正式四个 replay case

归档保留一训练、一验证、两测试 case 的原始请求、固定调度、拓扑、目标和链路证据。每案含两条预热请求和两条测量请求。正式分词回执保存输入长度及 token ID 摘要。

解压四案 ZIP，或将本目录的 cases 作为本地 case 根。部署方在远端实验根使用同样的 cases 子树。冻结适配器用 formal_case_set_manifest.json 与同目录的 formal_tokenization_receipt.json。父 case_set_manifest.json 保留六案的描述符，供冻结代码复核预登记索引。发布 ZIP 的成员 SHA 见公开 overlay。provenance 中的原传输清单覆盖六案，用于核对来源。

历史目标保留原 GPU UUID、序号、型号、能力与链路证据。原运行的私有 schema、启动配置和连接信息由操作方分别持有。迁移到其他硬件或模型时，保持请求与调度语义，重新绑定目标、分词、链路和资格，生成新的摘要。逐案重新分词可用同一历史模型的本地 tokenizer 核对，具体重分词状态见 overlay。

[四案 ZIP](formal-four-cases.zip) · [公开绑定](PUBLIC_CASE_OVERLAY.json) · [原六案传输清单](provenance/CASE_TRANSFER_MANIFEST.json) · [正式分词回执](cases/formal_tokenization_receipt.json)
