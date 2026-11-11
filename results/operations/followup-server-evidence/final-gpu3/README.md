# CUDA server evidence for final-gpu3

The server archive roster covers all 134 completed local leases in [the original result directory](../../../cuda-rsi-v41-gpu3/). Its artifact manifest SHA-256 is `7240095c37049f7012c9bf14bdd7b552640c87b59791110ea49e9433b2ba0b3f`. [The audit](index.json) records per-member hashes and request, result and process log checks.

Fully matched server records: 134.

Recheck from this directory.

```sh
python -B verify_topomem_cuda_followup_server.py --server-export . --results ../../.. --group final-gpu3
```
