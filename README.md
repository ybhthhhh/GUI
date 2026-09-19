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
