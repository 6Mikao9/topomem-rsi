# 复核工具

verify_sglang_publication.py 校验发布目录清单和租约回执 ZIP，可恢复供查看器使用的 loose receipts。verify_sglang_final_server_evidence.py 从服务器原件复算请求、摘要、全局调度和资格阶段的 packed KV。两个脚本都使用 Python 标准库，服务器脚本从冻结源码加载原调度校验器。

export_* 为本次封存使用的导出程序，要求操作方提供原终态及对应 SHA-256。test_* 保存本次运行的模拟测试源码，原工作区路径用于定位历史冻结文件和公开样例。发布目录的离线证据校验用 verify_*，模拟结果未纳入实测结果。
