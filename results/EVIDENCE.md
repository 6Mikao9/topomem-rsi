# Reading the measured evidence

The result folders preserve completed, rejected, failed, and interrupted experiments. A source revision being admitted does not establish a performance win over the native memory manager. Read each experiment's audit and scope before comparing numbers.

The CUDA v6 and v7 raw receipt files are published in `agent_tools/receipts.zip`. Their adjacent `receipts.archive.json` records the archive hash and every original file's byte count and SHA-256. The local workspace also retains the loose originals. Archiving does not alter receipt bytes or their embedded evidence hashes.

From an experiment's `agent_tools` directory, restore the receipt paths used by the report builder:

```sh
python -m zipfile -e receipts.zip .
```

Verify the archive's SHA-256 against `receipts.archive.json` before extracting. The self-contained HTML report can be opened without extracting any files. It embeds the evidence it displays; the raw receipts and journal allow independent reconstruction.

`source-snapshots/manifest.json` identifies the exact historical evaluator packages. A newer checkout is not a substitute for a historical source manifest. CUDA single-GPU engineering qualification, multi-GPU target configuration, actual measured KV placement, and model-serving experiments have separate scopes.
