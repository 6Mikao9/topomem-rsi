# Provenance and third-party materials

The controller, bounded memory-program language, evaluators, adapters, and evidence viewer in this repository are project implementations. The framework audit identifies the upstream research and implementations that informed the design, including their inspected revisions and licenses. Referencing a design does not imply that an upstream framework produced or validated this project's results.

Runtime dependencies retain their respective licenses. SGLang, PyTorch, CUDA, NCCL, and model weights are not redistributed as runtime binaries in this repository. Reproduction requires the versions and source hashes recorded in the experiment contracts. Deployable project source snapshots and measured receipts are versioned separately from those dependencies.

The upstream model card preserved under `results/sglang-qualification/model-provenance/` is attributed to [inference-optimization/GLM-5.3-0.6B-A0.4B](https://huggingface.co/inference-optimization/GLM-5.3-0.6B-A0.4B/tree/20aae340d157bd2215b9d81165be87e94686dfdf). Its metadata declares MIT. The recorded checkpoint was randomly initialized and trained on a small toy-text dataset; it is an architecture-testing model. This repository's receipts do not establish application quality or production-model locality.

No license grant for the project's original code is declared by this notice. Publication visibility and licensing are separate choices.
