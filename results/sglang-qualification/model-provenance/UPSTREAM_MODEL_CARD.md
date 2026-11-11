---
license: mit
base_model:
- zai-org/GLM-5.3
library_name: transformers
---

# GLM-5.3-0.6B-A0.4B

This is a tiny version of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) created for testing and development.

## Model Details

- **Base Model**: zai-org/GLM-5.3
- **Architecture**: glm_moe_dsa (GLM MoE with DeepSeek Sparse Attention)
- **Total Parameters**: 0.62B
- **Activated Parameters**: ~0.43B

This tiny model preserves the full GLM-5.3 architecture: Multi-head Latent Attention (MLA)
with low-rank Q/KV projections, the DeepSeek Sparse Attention (DSA) lightning indexer,
a mix of dense and MoE (sparse) MLP layers, routed experts with a sigmoid gate and
`e_score_correction_bias`, and a shared expert. Both `full` and `shared` indexer types
are represented.

## Configuration Changes

The following parameters were reduced from the original model:

| Parameter | Original | Tiny |
|---|---|---|
| num_hidden_layers | 78 | 6 |
| first_k_dense_replace | 3 | 1 |
| hidden_size | 6144 | 1024 |
| intermediate_size (dense MLP) | 12288 | 2048 |
| moe_intermediate_size | 2048 | 512 |
| n_routed_experts | 256 | 32 |
| num_experts_per_tok | 8 | 8 |
| n_shared_experts | 1 | 1 |
| num_attention_heads | 64 | 8 |
| num_key_value_heads | 64 | 8 |
| q_lora_rank | 2048 | 512 |
| kv_lora_rank | 512 | 512 |
| index_n_heads | 32 | 8 |
| index_head_dim | 128 | 128 |
| num_nextn_predict_layers (MTP) | 1 | 0 |
| vocab_size | 154880 | 154880 |

MLA head dimensions (`qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256`,
`head_dim=64`) and the indexer `index_head_dim`/`index_topk` are kept identical to the
base model to preserve the attention structure.

## Checkpoint Structure

Single-shard safetensors checkpoint (`model.safetensors`). The tensor naming is analogous
to the original GLM-5.3 checkpoint (`model.embed_tokens`, `lm_head`, MLA projections
`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`, the DSA `self_attn.indexer.*`
tensors, per-expert `mlp.experts.N.{gate,up,down}_proj`, `mlp.shared_experts.*`, and the
`mlp.gate` router). Two expected differences: this model is stored in bfloat16 rather than
FP8, so it has no `weight_scale_inv` scale tensors; and the multi-token-prediction (MTP)
layer is omitted (`num_nextn_predict_layers=0`), so it has no `eh_proj`/`enorm`/`hnorm`/
`shared_head` tensors.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("GLM-5.3-0.6B-A0.4B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("GLM-5.3-0.6B-A0.4B")

input_ids = tokenizer("According to all known laws", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

## Creation Process

This model was created using the llm-compressor `create-tiny-model` claude skill.

1. Inspected the `glm_moe_dsa` config to identify the layer/expert/attention controls.
2. Reduced layer count, hidden/FFN sizes, expert count, and attention/indexer heads to
   reach ~0.6B parameters while keeping at least one dense and one MoE layer and both
   indexer types. Weights were randomly initialized (with a fixup pass for norms/biases).
3. Fine-tuned on a small toy text dataset until perplexity converged well below target.
4. Verified the checkpoint tensor structure matches the base model (aside from FP8 scales
   and the MTP layer).
5. Validated loading, perplexity, and generation.

## Notes

- Stored in bfloat16 (the base model ships FP8-quantized weights).
- The MTP layer is intentionally omitted.
- Validation output:

```
Success: 1.0138424634933472 <= 10.0

==================================================
Generating sample text:
According to all known laws of aviation, there is no way a bee should be able to fly.
==================================================
```
