# CUDA server evidence for v13

The server archive roster covers all 24 completed local leases in [the original result directory](../../../engineering-producer-v13/). Its artifact manifest SHA-256 is `4d9e36cd23c2206135f950839a007405981ca24e6daa186968bb9c2671f4ef38`. [The audit](index.json) records per-member hashes and request, result and process log checks.

Fully matched server records: 24.

Recheck from this directory.

```sh
python -B verify_topomem_cuda_followup_server.py --server-export . --results ../../.. --group v13
```
