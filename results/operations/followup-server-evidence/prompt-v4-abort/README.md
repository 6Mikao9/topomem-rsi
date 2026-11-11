# CUDA server evidence for prompt-v4-abort

The server archive roster covers all 11 completed local leases in [the original result directory](../../../prompt-v4-abort/). Its artifact manifest SHA-256 is `b5e2d70cdefc441a4c6fa30eaa7aaadefe1933bed47081779cf491a403bc74d5`. [The audit](index.json) records per-member hashes and request, result and process log checks.

Fully matched server records: 11.

Recheck from this directory.

```sh
python -B verify_topomem_cuda_followup_server.py --server-export . --results ../../.. --group prompt-v4-abort
```
