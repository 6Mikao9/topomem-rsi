"""Exact installed SGLang sources bound by the topology adapter."""

from hashlib import sha256
from pathlib import Path


COORDINATOR = "srt/managers/hisparse_coordinator.py"
SOURCES = {
    "srt/mem_cache/sparsity/factory.py": "be8068d45f9422e5a6c7a6a122980011e67a08adb36753269dbe0ec87ee89bf3",
    "srt/mem_cache/kv_cache_configurator.py": "691bef2acb4a9dfbc815bac5d130723b1b0ec8b9cb45fd751d586b17a1b633a6",
    COORDINATOR: "4f5f8b0267797a32bc6a7d607379827d85ea958c1a6f50eadcde1d9aa6cda446",
    "srt/mem_cache/pool_host/mla.py": "868f42bfc75d28d0da376df4db075528c31c71aa1500462bc47b77a386bfe1d0",
    "srt/mem_cache/pool_host/base.py": "877cae9214ec1940160da84e97aeed3eacf5d25dc7bf2dc9aa9d0d420641efad",
    "srt/mem_cache/pool_host/common.py": "7b05da6ae2453c0af8dc040d12ffa84043d0e087c9ad52cec0063f9f42daa303",
    "srt/model_executor/model_runner.py": "df38e26ace5deae708f2353b56962afab7f92fac35d1bd4f274b7d6d62356372",
    "srt/layers/attention/dsa_backend.py": "312f8477a9ddb9fcb358595b6c986ee825e0a773d9e5d185224f37972bee5d8c",
    "srt/entrypoints/engine.py": "7d79f61f6171902d60dcece76ce6ee16e11c4dae6e9e98e620941be92effc6b0",
    "srt/managers/scheduler.py": "439ee96dd785fbb63a9f139b192cec984c252df16a573a436877f0ef9a4ca045",
    "srt/managers/schedule_batch.py": "06777e7b4c8a1d0b978a42c0370cf359a552fe7a23db411262f650db19755f7d",
    "srt/managers/scheduler_components/batch_result_processor.py": "e4fb82ba6bc501f2e8f29a0fc76ce240d1d128210122619c899917433c2b6ddb",
    "kernels/jit/csrc/kvcacheio/hisparse.cuh": "04e06f66b76fb21305b92161bce28e7412d5ffa63cc077179b6bb87751fbef12",
    "kernels/jit/csrc/kvcacheio/hicache.cuh": "eafe4522171ab8bdfec65e8565c8b0947c2854368fe6d7de1ad873043091eaa6",
    "kernels/jit/include/sgl_kernel/runtime.cuh": "dae99a49b116db59355c41a07495c78f9b0287e303316c07a7a63ce685eb5faf",
    "kernels/ops/kvcache/hisparse.py": "dbadcf1bbbf61af61b57f07f3b04ea805cf1caecfbf9b72b753268367e2ba415",
    "srt/environ.py": "c57c6ea499a47ed0bc80b299d2a6b224662bccb290af9083838b2b40aaa2adc0",
    "srt/layers/attention/dsa/dsa_indexer.py": "2f9634d275bde01c6eaffa0fabc339d592d2de67fa397b2e97af673f78ffbbfc",
    "srt/layers/attention/dsa/dsa_indexer_metadata.py": "9737568ff8222bf6ad302de81d05d30ae7c8227902e4f94d695d67575d2188ce",
    "srt/layers/attention/dsa/dsa_topk_backend.py": "d226b4579f302c295bce614e6df2525a9e1d79ba9dc3711d3f723e5b5627b203",
    "kernels/ops/attention/dsa/transform_index.py": "692d0aafb2236af1376c7fb77d3ba19e5dcbc72a398a159f8c6613a3dc4e8c31",
}


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source(path, expected):
    path = Path(path)
    if file_sha256(path) != expected:
        raise ValueError(f"SGLang source fingerprint differs for {path.name}")


def check_sources(sglang):
    if sglang.__version__ != "0.5.20":
        raise ValueError("topology memory requires SGLang 0.5.20")
    root = Path(sglang.__file__).resolve().parent
    for name, expected in SOURCES.items():
        check_source(root / name, expected)
    return root
