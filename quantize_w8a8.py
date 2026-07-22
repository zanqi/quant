import os
from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifers.quantization import QuantizationModifier
from huggingface_hub import HfApi

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-122B-A10B")
OUT_DIR = os.environ["OUT_DIR"]
REPO_ID = os.environ["REPO_ID"]

os.makedirs(OUT_DIR, exist_ok=True)

# TODO: MSE observer (data-free)?
#       activation dynamic (computed at run time
#           -> no calibration set)?
recipe = QuantizationModifier(
    config_groups={
        "int8_moe": {
            "targets": [r"re:.*language_model\.layers\.\d+\.mlp\.experts$"],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symetric": True,
                "strategy": "channel",
                "observer": "mse",
            },
            "input_activations": {
                "num_bits": 8,
                "type": "int",
                "symetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        }
    },
    ignore=[
        "lm_head",
        "re:.*embed.*",
        "re:.*self_attn.*",
        "re:.*linear_attn.*",
        r"re:.*mlp\.gate$",
        "re:.*shared_expert_gate$",
        r"re:.*mlp\.shared_expert\..*",
        r"re:.*visual\..*",
    ],
)

print(f"[1/4] Loading {MODEL_ID} (bf16, device_map=auto: 4xA40 + CPU)...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="bfloat16",
    device_map="auto",
)

print(f"[2/4] Running data-free W8A8 quantization (expect ~1-3h)...", flush=True)
oneshot(model=model, recipe=recipe, output_dir=OUT_DIR)

AutoProcessor.from_pretrained(MODEL_ID).save_pretrained(OUT_DIR)

# verify
import glob, json
idx_files = glob.glob(os.path.join(OUT_DIR, "*.index.json"))
expert_scales = 0
if idx_files:
    weight_map = json.load(open(idx_files[0]))["weight_map"]
    expert_scales = sum(1 for k in weight_map if "mlp.experts" in k and "scale" in k)

print(f"[verify] expert scale tensors written: {expert_scales}", flush=True)
if expert_scales == 0:
    raise SystemExit(
        "ERROR: no expert scales found -> llm-compressor did not unfuse the "
        "experts. Upgrade to llmcompressor>= 0.11 and rerun. Not pushed"
    )

# model card
card = f"""---
license: apache-2.0
base_model: {MODEL_ID}
tags:
- compressed-tensors
- int8
- w8a8
- vllm
- qwen3_5_moe
---
# {REPO_ID.split('/')[-1]}

INT8 (W8A8) quantization of [{MODEL_ID}](https://huggingface.co/{MODEL_ID}),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor).

**Scheme.** Routed MoE expert FFNs quantized to INT8: per-channel symmetric
weights (MSE observer) + dynamic per-token symmetric INT8 activations. Data-free
(no calibration set). Left in bf16: attention, GatedDeltaNet linear attention,
routers (`mlp.gate`, `shared_expert_gate`), the shared expert, embeddings,
`lm_head`, and the full vision tower.

**Why INT8 W8A8.** Targets INT8 tensor cores (compute capability >= 7.5, e.g.
A40/A100). The win shows up as ~2x matmul throughput on **text prefill** at
batch/concurrency; single-stream decode stays bandwidth-bound (INT8 weights are
the same byte size as FP8) so decode latency is roughly unchanged vs an FP8 build.

## Serving (vLLM)
```bash
vllm serve {REPO_ID} \\
    --quantization compressed-tensors \\
    --tensor-parallel-size 4 \\
    --max-model-len 16384 \\
    --reasoning-parser qwen3 \\
    --language-model-only
```
"""

open(os.path.join(OUT_DIR, "README.md"), "w").write(card)

# push
print(f"[3/4] Creating repo {REPO_ID}...", flush=True)
api = HfApi()
api.create_repo(REPO_ID, exist_ok=True, private=False, repo_type="model")

print(f"[4/4] Uploading {OUT_DIR} -> {REPO_ID}...", flush=True)
api.upload_large_folder(repo_id=REPO_ID, folder_path=OUT_DIR, repo_type="model")

print(f"DONE -> https://huggingface.co/{REPO_ID}", flush=True)