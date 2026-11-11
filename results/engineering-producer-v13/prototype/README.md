# v13 allocation-replica lowering

v13 derives from the frozen single-GPU v11 archive. Its source archive changes one file, `integrations/topology_memory/torch_runtime.py`. At initialization it caches the first eligible rule for a zero-hop producer retain only when the allocation-replica group has no dynamic features, has no admitted route to another destination, and has no competing allocation movement action. The producer copy must already reside on the local device when the shortcut runs. All ordinary moves to that existing destination are excluded by the original candidate filter. A numeric error during analysis leaves the rule on the original path, where it is raised only when evaluated.

The historical four-case LRU-equivalent receipt records 194, 228, 1,278 and 94 single-candidate retains. The CPU-only diagnostic measured the original and patched `_optional` call with the same modeled producer state and real timeline append, in 12 alternating blocks of 20,000 calls per arm. The medians were 2,419.37 and 236.315 ns per call, with a paired median saving of 2,187.355 ns. This measurement covers the modeled decision call. The frozen GPU study tests whether it moves end-to-end timing.

The preregistration binds GPU3, CPU64, four previously published training cases, three warmups, five samples and eight interleaved three-arm blocks. Native and LRU-equivalent v11 use the same original source. LRU-equivalent v13 uses the same policy and checks exact ordered actions, lifecycle, bytes, capacities, selected KV and outputs against v11. The expected counter change is 1,794 generic single-candidate retains replaced by 1,794 static retains per four-case diagnostic. This is an engineering experiment on the same policy. GPU data is pending.

Six CPU tests cover rule order, delayed numeric errors, dynamic features, pressure, competing routes and actions, and the 24-job protocol against historical receipt shapes. The protocol mock also verifies rejection of a changed action.

- Plan `gpu3-paired-study-r1/preregistration.json` SHA-256 `19789ad0cd6c55cd3faeed4aab151854f3c0c1f8608c9ccd173ccdfe90dc9e29`
- Source `gpu3-paired-study-r1/v13.source.zip` SHA-256 `8d3a98e1791fdc4fb40564e1341686de466b6e8105d033c7fd7f97459eee52f6`
- Source manifest SHA-256 `e66bf6796bfec6e14a5980bd8effcab96dd5ba4da7a2ba339103ae9f40e93f5d`

After the operator stages and verifies the frozen v13 archive, use the previously qualified GPU3 executor and frozen v11 local controller.

```powershell
python -B qualify_v13_paired_gpu3.py run --plan gpu3-paired-study-r1/preregistration.json --expected-plan-sha256 19789ad0cd6c55cd3faeed4aab151854f3c0c1f8608c9ccd173ccdfe90dc9e29 --executor-config PATH_TO_PRIVATE_GPU3_EXECUTOR_CONFIG --controller-root PATH_TO_FROZEN_V11_CONTROLLER --v13-source src-v13-producer-retain-engineering --audit-directory PATH_TO_NEW_RUN_DIRECTORY
python -B qualify_v13_paired_gpu3.py analyze --plan gpu3-paired-study-r1/preregistration.json --expected-plan-sha256 19789ad0cd6c55cd3faeed4aab151854f3c0c1f8608c9ccd173ccdfe90dc9e29 --audit-directory PATH_TO_NEW_RUN_DIRECTORY --output PATH_TO_NEW_ANALYSIS_JSON
```
