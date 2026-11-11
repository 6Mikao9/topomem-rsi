# Training-only SGLang hand-policy sweep

This document records the original eight-policy design. The completed resident v2 sweep adds a ninth strategy and uses its own pinned source, result, and receipt archive. [Resident v2 protocol](SG_RESIDENT_V2_ZH.md) · [Completed training sweep](../results/sglang-hand-sweep-v2/README_ZH.md)

The fixed 6,144-slot all-host stock control can retain the declared short contexts, so moving a cold slab to HBM may only change writeback on this workload. The sweep tests whether that helps real serving time. It leaves the declared stock comparison in the recursive experiment intact.

The eight hand policies are the 4,096-slot all-host default, the 6,144-slot all-host reference, and six 6,144-slot policies that each keep one same-index layer in local HBM on both attention ranks. Other layers stay in pinned host memory. The sweep uses the measured hot metadata profile and the same topology, KV budgets, fixed request schedule and inputs. It rejects a candidate if its requested pair of local slabs cannot fit. Each slab is 21,537,792 bytes. Under the 25,715,808-byte per-rank incremental HBM limit, a slab fits if the observed hot metadata increase is at most 4,178,016 bytes on that rank. The 6,144-slot all-host control stays in the sweep even when local slabs fit.

The two stock policies run in native mode. Each capacity first passes two fresh native parity runs per training case. Each slab policy then needs parity with its matching native capacity before timing. Performance trials use post-warmup replay time. A policy displaces the 6,144-slot reference only if it is faster in every paired block and its median gain meets the declared practical margin. Pair scores sum the individual replay windows across the fixed training cases. The default three blocks are an incomplete prefix of the balanced eight-policy order. Eight blocks cover a full cycle when all policies qualify. With one training case and all eight policies, the three-block sweep takes at most 34 fresh GPU jobs. Existing bound parity receipts can be reused by the leased worker. Actual elapsed time depends on model startup, cache state and replay time per job.

Preflight checks the bound training fixtures and profile without starting an executor. Use a new output directory and a unique sweep ID.

```sh
python -B -m topomem_sglang_rsi.manual_sweep \
  --adapter-config /path/to/adapter.json --output /path/to/new-preflight \
  --sweep-id hand-control-v1 --blocks 3 --minimum-gain-ms 5
```

Review the manifest and retain its `sweep_sha256`. To run the fixed matrix after leasing the qualified GPUs, repeat the same options with a new output directory and `--execute PREFLIGHT_SHA256`. The adapter writes its ordinary leased-job receipts, and the sweep writes the per-case timings, qualification job IDs and selected program. Keep a separate local receipt directory for this sweep. The measured selection uses training cases only. Any later use as a comparator needs a new frozen experiment or a separately labeled evaluation of the frozen hand program. This workload uses the pinned toy GLM checkpoint described in the [SGLang adapter notes](SG_CONFIGURABLE_RELEASE.md).

## Formal comparison with the training-selected control

Run the three-block, 5 ms hand sweep from the frozen schema-2 source before starting the model search. Preserve its original schema-2 adapter config, output directory, and leased receipts. For a new formal run, use `make_config` with `--manual-sweep RUN_DIRECTORY --manual-sweep-config ORIGINAL_SCHEMA2_CONFIG`. It creates schema 3 after reconstructing the full selection from archived training jobs, including native A/A, every feasible timing, the selected physical policy and the training-only calibration. The schema-3 contract pins source and artifact hashes, the selected program, and the receipt inventory. It rejects missing receipts and mismatches before model calls. Ordinary schema-2 configurations use the predeclared stock control.

The formal protocol fixes two rounds, two slots, at most five API calls per arm per round, 65,536 output tokens per call, high reasoning, six validation blocks, twelve sealed test blocks, and a 14,400-second run wall budget. The minimum paired improvement in milliseconds is the larger of 5 and twice the range of the three default-native training block totals. Use the preregistered launcher, which passes this fixed configuration to the controller.

Native probe jobs took roughly 50 seconds apiece after initial compilation. Before any SGLang model search or held-out evaluation, the cost plan selected the first listed training case and the first listed validation case from the original six-case manifest. Both test cases remain in order. The second training and validation cases are recorded as unused in search. This is a positional rule fixed from job cost. Preserve the original case set and tokenization receipt, and bind the derived case set and its token commitments to both original digests. The selected case descriptors, token IDs, validation block count, and held-out block count stay fixed. A change to the external schedule requires separately rebound case and schedule identities.

```sh
python -B -m topomem_sglang_rsi.manual_selection run \
  --adapter-config /path/to/schema3.json --output /path/to/new-run \
  --key-file /path/to/local/key-file
```

After the run, use the same adapter and output with the `freeze` command and then the separate `test` command. Both commands run with API access disabled. The hand-sweep evidence and model-driven results are distinct. The preregistration specifies how the comparison will be run once the GPU qualification and training sweep finish.
