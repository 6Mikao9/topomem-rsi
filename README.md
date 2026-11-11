# TopoMem RSI

[中文说明](README.zh-CN.md) · [Two-task architecture (中文)](docs/ARCHITECTURE_ZH.md) · [Memory program contract](docs/MEMORY_PROGRAM_CONTRACT_ZH.md) · [SGLang reproduction (中文)](docs/REPRODUCE_SGLANG_ZH.md) · [Offline report](examples/topology-report/README.md)

TopoMem RSI searches for KV memory programs for a fixed sparse-attention implementation, hardware topology, and workload. The controller can also revise the code that generates those programs and the code that proposes the next revision.

## Start here

1. **Open the final offline demo.** Download the HTML and open it in a local browser with JavaScript enabled. The SGLang held-out result is selected on load. Use the run selector to open the CUDA Agent evidence. [Offline demo](demo/topomem-evidence.html)
2. **Run a local CPU check.** From the repository root, use Python 3.10 or later. The following tests use the standard library and work directly from the checkout.

   ```sh
   python -B -m unittest discover -s tests -p 'test_memory_program.py'
   python -B -m unittest discover -s tests -p 'test_topology_report*.py'
   ```

3. **Audit archived measurements offline.** This command verifies the engineering archive and member hashes without extracting files. It reports `verified_members` for four archives. The linked audit also documents how to reconstruct the measured summaries. [Engineering archive and audit](results/engineering-calibration/README.md) · [SGLang sweep audit](results/sglang-hand-sweep-v2/REPRODUCE_ZH.md)

   ```sh
   python -B results/engineering-calibration/extract_receipts.py --manifest results/engineering-calibration/receipts-manifest.json
   ```

For GPU execution, follow [the CUDA setup](#optional-cuda-gpu-execution) or the [SGLang deployment guide](docs/REPRODUCE_SGLANG_ZH.md). The SGLang demo replays a tiny GLM architecture checkpoint that was randomly initialized and then fine-tuned on toy text. The observed outputs are EOS IDs. This replay measures fixed-request execution and KV correctness. Production throughput and answer quality require separate evaluation. [Model provenance](results/sglang-qualification/model-provenance/UPSTREAM_MODEL_CARD.md)

## Two tasks

| Task | Output | Evidence |
| --- | --- | --- |
| Generate a memory manager | A bounded `MemoryProgram` with event rules for placement, replication, migration, pressure-driven demotion, prefetch, eviction, and reclamation | Executed actions and data-movement receipts, byte and ownership checks, and paired timing of the resulting program against a frozen reference |
| Improve the generator | Versioned `build_prompt` and `build_meta_prompt` source and a structured improvement plan | Matched parent–child decisions and results, an admission decision, and subsequent task and meta requests bound to the admitted source digest |

The task entry chooses an operator, parent programs and training evidence. The meta entry chooses a revision strategy and training evidence. Its model call returns `source` and `plan` in one response. The controller binds the proposed source digest and the plan's observed problem, mechanism change, expected effect and falsification condition before evaluating the child. The expected effect is a hypothesis. The next round's actual task and meta calls provide separate evidence of reuse.

The target fixes attention and token selection, the physical GPU identities and links, capacity, compute ranks, request arrival and ordering, lifetimes, batching, and safe points. Generated code sees training evidence and chooses memory actions. The evaluator owns correctness and scoring. The controller freezes the finalists before opening the test split.

## Status

Paired blocks repeat the fixed experimental arms in separate jobs. Generator runs and source admissions are counted separately.

### Formal SGLang v3 experiment

The two-round experiment admitted zero generator revisions. The frozen adaptive program came from the original generator in R0. It used 4,096 hot slots and placed layer 0's cold KV in local HBM on each rank. The registered held-out comparison used 12 paired blocks and two cases per block. Gain is the comparator's replay time minus the adaptive program's replay time. The gate requires a positive gain in every block and a median gain above 11.492 ms.

| Comparator | Adaptive faster | Median paired gain | Registered gate |
| --- | ---: | ---: | --- |
| This experiment's native 4,096-slot control | 7/12 | +10.958 ms | Failed |
| Fixed generator | 0/12 | −32.569 ms | Failed |
| Training-selected hand policy | 0/12 | −26.065 ms | Failed |

The frozen fixed-generator program selected `resident-6144`, matching the recorded memory configuration of the hand policy. In a descriptive comparison from the same blocks, the fixed program beat the native 4,096-slot control in 12/12 blocks with a +35.585 ms median paired gain. That program came from the unchanged generator. [Independent audit (中文)](results/sglang-rsi-v3/independent-audit/TERMINAL_AUDIT_ZH.md) · [Formal controller report](results/sglang-rsi-v3/controller/report.json) · [Frozen program selection](results/sglang-rsi-v3/selected/selection.json)

A separate preregistered comparison used the same frozen adaptive program against native 6,144-slot all-host backing. It was faster in 5/8 blocks with a +11.923 ms median paired gain and failed the all-blocks gate. The adaptive arm used 4,096 slots, so this comparison varies both hot capacity and backing placement. [Additional comparison](results/sglang-native6144-ablation-v1/README_ZH.md)

### Earlier experiments

- **CUDA v6 Agent diagnostic.** One round settled before the run was stopped to correct the generation protocol. Its child changed both function ASTs and made the same structural decisions on that round's matched inputs. It was rejected. Across 12 paired validation blocks the median child score was 146.529 ms and the median write-through/LRU reference score was 120.898 ms. The reference was faster in all 12 blocks. All five measured arms had zero managed KV on the second GPU. The local audit covers 66 result receipts and preserves the partial next round. [v6 audit](results/cuda-rsi-v6/agent_tools/AUDIT_R0_ZH.md)
- **CUDA v7 Agent search.** All six DeepSeek rounds and 12 held-out blocks settled. R2 was the sole source admission. Its program recovered earlier behavior and beat the fresh parent in 12/12 validation blocks, with a +126.658 ms paired median gain, while losing all 12 to the CUDA write-through/LRU reference. The admitted source then drove nine actual task and meta requests across R3–R5. The frozen adaptive manager came from the original generator in R0. It lost 12/12 held-out blocks to the CUDA reference, with a −29.172 ms paired median gain, and won 6/12 against the fixed generator, with a −0.142 ms paired median gain. The source admission and reuse are distinct from the held-out manager result. [v7 audit](results/cuda-rsi-v7/agent_tools/AUDIT_PROGRESS_ZH.md)
- **CUDA v11 four-scenario transfer.** The frozen seed and v7 R2 generators produced programs for Agent, chatbot, long-document QA, and rollout. All 192 held-out jobs passed the recorded KV, attention-output and ownership gates. The admitted source lost all 48 paired blocks to the project's CUDA write-through/LRU reference. Against the seed, chatbot was faster in 12/12 blocks, but only 11/12 exceeded its 1.5 ms margin. None of the four scenarios met the strict transfer gate against the seed. Observed managed KV usage on the second GPU was 0 B in all four arms. [Transfer audit](results/cuda-transfer-v11/audit/AUDIT_ZH.md)
- **SGLang resident v2 native qualification.** The exact v2 serving source passed six fresh TP2/attention-DP2 native runs on the public calibration replay, two each at 4,096, 6,144 and 8,192 slots. Both ranks agreed on selected packed KV, output IDs, dispatch and 132 active global forward ticks. The adapter can select startup hot capacity, static cold layer placement, or full residence under the case proof. [Native qualification](results/sglang-qualification/ingress-resident-v2/AUDIT_ZH.md) · [Resident v2 deployment and protocol](docs/SG_RESIDENT_V2_ZH.md)
- **SGLang nine-policy hand sweep.** All 38 jobs settled on one formal training case, including 27 timing jobs in three blocks. The hand-written `resident-6144` was selected, with a +22.234 ms paired median gain against native 6,144 and +25.187 ms against this experiment's native 4,096-slot control. The spread of the 4,096-slot control's training windows gives an 11.492 ms formal threshold. The formal v2 search stopped after a prompt-contract conflict between the unconditional `allocate/place` requirement and the resident mode. The training sweep supplies the manual control for the formal comparison. [Training audit](results/sglang-hand-sweep-v2/README_ZH.md) · [Raw selection](results/sglang-hand-sweep-v2/sweep/result.json) · [Training case binding](results/sglang-hand-sweep-v2/training-case/manifest.json)
- **Independent resident v3 engineering probe.** On the fixed public tiny-GLM replay, two stock/resident timing pairs showed 28.003 ms and 28.479 ms shorter windows with the human-engineered full-resident primitive. Per-rank cold host payload fell from 129,226,752 B to 0 B and the six CUDA sentinel rows used 3,936 logical bytes. These observations motivated adding the primitive to the nine-policy sweep. The toy checkpoint was randomly initialized and then fine-tuned on toy text, and this replay emitted EOS IDs. [Engineering probe audit](results/sglang-qualification/resident-v3/README.md)
- **Stock SGLang probes.** A TP2 backbone with attention DP2 verified one routed request per attention rank and eight local query heads on each rank. A separate stock UnifiedRadixCache probe verified 1,536 physically shared prefix rows and disjoint 128-row suffixes for each of two branches. The recorded cached and isolated outputs in the radix fixture consist of EOS IDs. [SGLang qualification evidence](results/sglang-qualification/evidence_manifest.json)

## Optional CUDA GPU execution

Use a local Python 3.10 or later environment for the controller. An editable install is optional when running from the repository root. The remote CUDA evaluator needs a CUDA-enabled PyTorch environment. For model serving use the separately pinned SGLang stack. [SGLang reproduction](docs/REPRODUCE_SGLANG_ZH.md)

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -B -m unittest discover -s tests
```

Build a reproducible source-only CUDA deployment bundle.

```sh
python -B -m rsi_memory.source_bundle --output deploy/cuda-source.zip
```

Unpack it into a fresh source directory under the worker's experiment root. Pin the reported `source_manifest_sha256` in the experiment input. The manifest covers the archive's source files and fixtures by relative path. On the target GPU host, measure its directed pinned-host copies using `python -B -m integrations.topology_memory.torch_profile --output link-profile.json`. Configure the execution client with the target's SSH configuration, worker and Python locations, GPU ordinals and UUIDs, and pinned source. Keep local configuration and provider credentials outside the shared source and reports.

Create `private-build-inputs.json` with the measured link profile, topology, fixed workloads and control programs. The [local input table](README.zh-CN.md#准备本地配置) describes every required field. The command writes one sealed config per scenario, including `sealed-configs/agent_tools.json`.

```sh
python -B -m rsi_memory.experiment_cases --inputs private-build-inputs.json --output sealed-configs
```

## Run, freeze, test

In Bash, point `args` at a new output directory and one sealed scenario config. Discovery makes model API calls. It reads `OPENCODE_GO_API_KEY` locally, or accepts `run --key-file /path/to/local-key-file`. The GPU worker receives the sealed inputs and programs. `freeze` is local and `test` uses GPU evaluation with the model API disabled.

```bash
args=(--output runs/agent_tools/controller
      --adapter rsi_memory.cuda_domain:factory
      --adapter-config sealed-configs/agent_tools.json
      --model deepseek-v4.1-flash --rounds 6 --slots 2 --max-output-tokens 65536
      --min-paired-blocks 12 --min-improvement 0.5 --wall-seconds 43200)
python -B -m rsi_memory.topology_rsi run "${args[@]}"
```

After discovery settles, review the journal and freeze the primary and fixed-generator finalists. Then run the separate held-out test with the same configuration.

```bash
python -B -m rsi_memory.topology_rsi freeze "${args[@]}"
python -B -m rsi_memory.topology_rsi test "${args[@]}"
```

The journal keeps failed and interrupted attempts. Resuming requires the same source and run contract. An attempt with an unknown outcome is preserved as unknown. The primary finalist comes from the seed or admitted generator lineage. The unrestricted archive best is recorded separately at validation scope.

## Inspect actual evidence

From the repository root, rebuild an offline view of the archived six-round journal, four-scenario transfer, SGLang training sweep and engineering audits. Choose a fresh output path. The renderer protects existing HTML by requiring a new filename.

```sh
python -B -m rsi_memory.topology_report \
  --run Agent=results/cuda-rsi-v7/agent_tools/controller \
  --transfer results/cuda-transfer-v11 \
  --manual-sweep results/sglang-hand-sweep-v2/sweep \
  --engineering results/engineering-calibration/engineering-v11-gpu3-qualification.json \
  --engineering results/sglang-qualification/resident-v3/audit.json \
  --output demo/rebuilt-archive.html
```

Open the generated HTML in a browser. Add original GPU receipts when you need to inspect the actual layouts and job evidence. [Receipt archive and extraction instructions](results/EVIDENCE.md)

Point the viewer at the run directory and its locally recorded execution receipts. The viewer writes one offline HTML file and reads the journal in read-only mode.

```sh
python -B -m rsi_memory.topology_report \
  --run Agent=runs/agent_tools/controller --receipts runs/agent_tools/receipts \
  --scenario examples/workloads/agent-water-inspection.json \
  --scenario examples/workloads/long-document-estuary-qa.json \
  --scenario examples/workloads/chatbot-trip-planning.json \
  --scenario examples/workloads/rollout-bridge-options.json \
  --output agent-evidence.html
```

Use `--transfer` for a separate transfer-study directory, `--manual-sweep` for the archived training selection, and `--engineering` for an engineering audit JSON. The viewer keeps measured results, preregistration, and pending phases labeled separately.

The CUDA evaluator executes text-derived block-sparse attention with real CUDA allocations and data movement. Its `native_lru` arm is the project's write-through/LRU reference. Its objective sums each case's median elapsed milliseconds, including immutable block materialization during decode appends and any resulting whole-block moves. The authored Agent, long-document QA, chatbot, and rollout fixtures test different lifecycles. The long-document fixture uses a LongBench-style task form. Report CUDA evaluator and model-serving results under their own scopes. [Workload contracts](examples/workloads/README.md) · [Framework audit](docs/RSI_FRAMEWORK_REEVALUATION_20261110_ZH.md)
