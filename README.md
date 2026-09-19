# GUI Memory Specialization — Stage 1

This repository tests one falsifiable claim: a history representation trained for
one frozen GUI policy may transfer worse to another frozen policy, even when both
consume the same canonical interface and budget.

It deliberately does **not** implement tool routing, model routing, free-form
summary prompting, latent-memory exchange, or a unified conditioned encoder.

## Staged protocol

1. **Interface and data validation (implemented):** validate action-aligned GUI
   trajectories and render fixed-budget canonical event slots.
2. **Offline cross-transfer pilot (implemented):** connect three frozen-policy
   adapters and evaluate every producer/consumer pair on the same held-out rows.
3. **Selective training (adapter-dependent):** train one budgeted selector per
   frozen consumer from action likelihood/reward; do not update the consumer.
4. **Online confirmation (not implemented):** replay promising cells in an
   execution-based GUI benchmark.

No toy result from this repository is scientific evidence. The included toy
adapter only verifies matrix bookkeeping and budget invariants.

The implemented baseline selectors are `no_history`, `recent`, `full_history`,
`random_events`, and `structured_heuristic`. `full_history` is explicitly marked
as an unmatched-budget upper-context control; every other selector uses the same
event budget.

## Canonical interface

Every event contains `pre_state -> action -> outcome -> post_state` plus bounded
visual-evidence references, subgoal, failure status, and temporal links. The
renderer is deterministic JSON with a fixed `max_events` budget. It contains no
model-specific wording.

For a screenshot-capable frozen policy, add an optional `current_screenshot`
absolute path to the sample. `Qwen25VLPolicy` uses the current screenshot only as
the policy observation; history remains the identical canonical interface used by
all consumers.

## Dataset format

One JSON object per line:

```json
{"sample_id":"s1","task":"Upload the saved report","current_observation":"upload dialog","gold_action":"type:/tmp/report.xlsx","history":[{"timestamp":1,"pre_state":"editor","action":"save_as","outcome":"success","post_state":"save dialog","visual_evidence":["filename field"],"subgoal":"save report","failure_status":"none","temporal_links":[]}]}
```

Use file references or stable visual-region IDs in `visual_evidence`; do not
inline screenshots in this first-stage text interface.

## Commands

```bash
python -m gui_memory_specialization.cli validate examples/toy_trajectories.jsonl
python -m gui_memory_specialization.cli smoke examples/toy_trajectories.jsonl
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:/home/work/cybergym/pylibs:$PYTHONPATH \
  python3 -m gui_memory_specialization.cli policy-smoke examples/toy_trajectories.jsonl \
  --model-path /share/fshare/common/models/Qwen/Qwen2.5-1.5B-Instruct --dtype float32
python3 examples/make_synthetic_gui.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:$PYTHONPATH \
  python3 -m gui_memory_specialization.cli vl-smoke examples/synthetic_vlm_trajectory.jsonl \
  --model-path /home/work/Qwen2.5-VL-7B-Instruct --dtype bfloat16
python -m unittest discover -s tests -v
```

`smoke` uses deterministic fake policies and must never be reported as a GUI
benchmark result. Replace them with adapters exposing `score_gold_action` before
any research claim.

`policy-smoke` is a single-row, single-GPU infrastructure check for the frozen
causal-LM adapter. It does not contain screenshots and is expressly excluded from
the GUI experiment's results table.

`vl-smoke` uses a generated mock GUI screenshot only to exercise the local
Qwen2.5-VL adapter. It is not a benchmark trajectory and is excluded from every
scientific result or ablation table.

### Mem-W web-stage infrastructure checks

The web-stage replacement for the blocked OSWorld rollout starts with a real
browser interaction and a real vision-model update.  These commands are
infrastructure checks only; they neither implement Mem-W's full training method
nor produce a scientific score.

```bash
# In an environment with Playwright plus Chromium installed.
python3 examples/check_memw_web_gui.py --proxy http://127.0.0.1:7897 \
  --screenshot outputs/memw-web-smoke.png

# On a GPU node with Qwen2.5-VL, Transformers, PEFT and its vendor PyTorch.
CUDA_VISIBLE_DEVICES=0 python3 examples/qwen25vl_lora_train_smoke.py \
  --model-path /home/work/Qwen2.5-VL-7B-Instruct \
  --image examples/synthetic_upload.png
```

The first test must report `MEMW_WEB_GUI_INTERACTION_OK`, proving browser GUI
observation, text entry, and navigation.  The second must report
`QWEN25VL_LORA_TRAIN_SMOKE_OK`, proving a screenshot-conditioned forward pass,
backward pass, and LoRA optimizer update.  Neither output validates online task
success: that requires a fixed task set and a programmatic success verifier.

### Formal continuous-memory protocol

The formal web experiment has two intentionally separated stages, following
the relevant Mem-W/CoMEM design rather than treating logged outcomes as online
rewards.

1. **Stage one — offline continuous-memory distillation.** The Qwen2.5-VL
   visual policy is frozen. A shared-weight, 1024-dimensional Q-Former
   compresses up to three prior screenshot/action observations into eight
   continuous tokens. The student sees those tokens plus the current webpage;
   the teacher sees the uncompressed history. The trainable compressor minimizes
   action cross-entropy plus KL divergence from the teacher at the same action
   token positions.
2. **Stage two — outcome-aware online update.** Only actual webpage rollouts
   with `source="online_execution"` are accepted. Sibling rollouts of one task
   form an RLOO group; their terminal rewards provide the policy-gradient
   advantage. Offline trajectory files, including their `result.txt` values,
   are rejected by this stage.

The first-stage corpus is built only from CoMEM archive members in `success/`.
It retains an episode-stable validation split and materializes only images used
by selected training positions:

```bash
python3 examples/prepare_comem_stage1.py \
  data/comem_raw/shopping.zip data/stage1/shopping_success.jsonl \
  --image-root data/stage1/images --max-samples 15000

CUDA_VISIBLE_DEVICES=0 python3 examples/train_memw_stage1.py \
  data/stage1/shopping_success.jsonl outputs/stage1-shopping \
  --model-path /home/work/Qwen2.5-VL-7B-Instruct \
  --max-steps 15000 --checkpoint-every 100
```

`stage1_last.pt` contains the compressor, optimizer, and completed step. Resume
without repeating completed updates using `--resume stage1_last.pt`. Before a
long run, `examples/smoke_memw_stage1_qwen.py` and the one-step launcher smoke
must pass on the exact node/runtime. The resulting stage-one checkpoint is a
training artifact, not a claim of task success; stage two requires executable
web tasks and terminal rewards from those executions.

### OSWorld-Verified rollout-host preflight

Before installing VM images or starting an online confirmation, run:

```bash
python3 examples/check_osworld_host.py --json
```

For the OSWorld Docker provider, `kvm_device`, `docker_cli`, and
`docker_daemon` must all be `true`. This preflight makes no system change and
never starts a VM. A nonzero exit code means the host is not ready for an
execution-based rollout; do not treat offline likelihood results as task success
until this check passes and a real benchmark smoke run succeeds.

### 31571 runtime note

On node 31571, the default Transformers 4.51 installation imports an incompatible
vendor `flash-attn`. The node-local, read-only Transformers 4.44.2 path above is
known to load the shared Qwen2.5-1.5B model. Preserve the existing `PYTHONPATH`
when adding `src`; replacing it hides the vendor PyTorch installation. The FP32
smoke configuration is intentional: FP16 produced non-finite scores and the
adapter now rejects them.
