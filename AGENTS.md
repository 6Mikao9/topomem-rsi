# Working on this repository

- Treat sparse attention, hardware and topology, workload events, scheduling, admission, batching, and rank placement as fixed inputs. Generated programs may change only the explicitly supported memory actions.
- Keep the memory-program task separate from generator self-improvement. Register each proposed generator change before evaluation. Preserve its source, paired comparison, acceptance decision, and subsequent task and meta calls. Human changes to the evaluator or available primitives are engineering changes, not recursive model improvement.
- Preserve frozen sources, case identities, unsuccessful experiments, and raw receipts. Do not edit a source tree used by an active run. Do not use validation or heldout results to select a training candidate, change a registered threshold, or retry an unknown external effect.
- A configured second GPU is not evidence that a policy used it. Report actual per-device KV occupancy and link traffic. Keep logical managed bytes, allocator allocation, and allocator reservation separate.
- Read the experiment-specific reproduction instructions before using a remote GPU. Reuse a persistent SSH connection to an A100 machine; frequent reconnects can trigger its access controls. Coordinate GPU leases through one owner and clean up only owned process groups.
- Use Sol for delegated coding and research agents. Do not launch Astra for simple tasks on a remote Codex instance.
- Credentials and machine connection configuration stay outside version control. Do not print keys, copy API credentials to GPU hosts, or publish private executor configuration.
- Run checks appropriate to the affected implementation. Offline fixture tests and engineering probes must remain clearly separate from measured GPU and model-generated evidence.
