# Stage-one model-specific compressor baselines

These safetensors contain only the trained compressor parameters from the
Mem-W-style stage-one runs. The frozen backbone models, optimizer states and
training data are not included. Matching JSON files record the checkpoint
step, training configuration, parameter count, and SHA-256 digests.

| Backend | Checkpoint step | Parameters | Notes |
|---|---:|---:|---|
| Qwen2.5-VL | 1151 | 19,951,616 | Four-GPU historical run; its image/history settings differ from the other two. |
| LLaVA-NeXT | 1149 | 21,001,216 | Eight-GPU run, 224 px / 2200-character cap. |
| InternVL2 | 1149 | 21,001,216 | Eight-GPU run, serialized gradient synchronization, 224 px / 2200-character cap. |

They are reproducibility baselines, **not** a valid cross-model transfer
interface: the output tokens live in each downstream model's embedding space.
Storyline's producer-by-consumer 3×3 study requires a common, structured
trajectory interface and separately trained compatible producers.

The binaries use Git LFS. Run `git lfs pull` after cloning if Git has left
pointer files in place. Export provenance: `examples/export_stage1_compressors.py`.
