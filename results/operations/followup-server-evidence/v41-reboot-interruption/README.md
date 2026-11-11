# CUDA server evidence for v41-reboot-interruption

The server archive roster covers all 26 completed local leases in [the original result directory](../../../cuda-v41-reboot-interruption/). Its artifact manifest SHA-256 is `d2030731dadc5322f28cf033d24c922a8c20ac17420ee3cc8a0f453f743174df`. [The audit](index.json) records per-member hashes and request, result and process log checks.

Fully matched server records: 25.

The reboot export contains one completed local lease with empty server result and receipt files. The original files are preserved. Its server input, request, started record and process log match the local lease.

Recheck from this directory.

```sh
python -B verify_topomem_cuda_followup_server.py --server-export . --results ../../.. --group v41-reboot-interruption
```
