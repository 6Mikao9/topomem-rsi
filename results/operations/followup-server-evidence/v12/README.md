# CUDA server evidence for v12

The server archive roster covers all 24 completed local leases in [the original result directory](../../../engineering-dynamic-v12/). Its artifact manifest SHA-256 is `bb0b65e899a2ce2ae7c48209dfc4a3d81ad92fd8907b14e5041cc693762920ef`. [The audit](index.json) records per-member hashes and request, result and process log checks.

Fully matched server records: 24.

Recheck from this directory.

```sh
python -B verify_topomem_cuda_followup_server.py --server-export . --results ../../.. --group v12
```
