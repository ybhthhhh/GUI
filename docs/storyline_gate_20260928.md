# Storyline canonical-memory gates (2026-09-28)

This is the first *producer × consumer* experiment, not the prior 3 models ×
3 history conditions. The earlier Q-Former checkpoints are retained as
model-specific Mem-W-style baselines, not cross-transferable producers.

## Fixed interface

All three frozen GUI VLMs score the same CoMEM shopping/service episodes and
the same next-action command. Each candidate memory contains exactly one observed
transition: a 224×224 letterboxed pre-state screenshot, its recorded action,
and the adjacent 224×224 post-state screenshot.
The current screenshot is a third, identical 224×224 image. The prompt field
order, task text, 2200-character target-action cap and episode-held-out split
are shared. Each model still uses its native tokenizer and vision tower, so
the number of *internal* visual tokens is not assumed equal; action-token
counts are recorded per candidate, and visual-token counts need a separate
adapter audit before any compute-matched claim.

The first **raw-response pilot** contains 3,518 training decisions and 370
validation decisions (3,888 total). An action-schema audit then found that
CoMEM `response` often includes natural-language reasoning; truncating prior
responses to 512 characters can cut the JSON command in half. This pilot is
retained as a wording/control comparison, not accepted as the strict gate.

The **structured-action gate** is rebuilt from the untruncated source. It
extracts the executable JSON `name` and operational arguments, discarding
free-form reasoning and descriptions. Rows with unparseable or overlong
commands are excluded rather than truncated. This leaves 3,146 training and
352 validation decisions, from 962 and 110 disjoint episodes. Exclusions are
1,164 one-event histories, 181 unparsed targets, 253 unparsed histories,
and 2 long targets. A candidate is an event index, not a model-specific latent.

The source lacks explicit step outcomes and counterfactual task rewards. The
only non-null outcome field is measured color change between observed before
and after screenshots; `success`, `failure_status`, and `subgoal` are unknown.
This prevents falsely treating the logged next action as an optimal policy.

## Downstream objective and transfer

Each frozen consumer evaluates every train/validation candidate's recorded
next-action cross-entropy (CE). Eight independent GPU workers per model cache
these losses without NCCL or modifying the VLM. A small, independent selector
`E_i(H,o_t)` is then trained for each consumer by minimizing expected *train*
candidate CE. Selector features use only current and prior observed visual
statistics, action type/length, transition magnitude, and temporal position;
the target action is unavailable to the selector. Calibration is separated by
training episode. Validation CE is never used in fitting or checkpoint choice.

For the 352 structured-action held-out decisions, each `E_i` selects one event. All three frozen
consumers then score that same event, producing the actual 3×3 matrix. Compare
within each consumer column: lower matched CE than transferred CE would
support, but not by itself prove, model-dependent memory specialization.
The matched-vs-transferred difference receives an episode-resampled 95% CI.
Recent-one, oldest-one, and largest-observed-change are fixed baselines.

This is an **offline action-likelihood gate**. It does not measure executed GUI
task success, failure recovery, or genuine reward. A positive matrix must be
followed by wording/field controls and online
execution before advancing to shared latent or model-conditioned memory.

## Live run placement

The raw-response pilot source and canonical manifest live under
`/home/work/gui-memory-specialization/validation/storyline-gate-20260928`
and `/home/work/gui-memory-specialization/data/storyline_gate`. Its scorer
outputs are under `validation/storyline-gate-20260928/formal/<backend>`.
The strict structured-action source is isolated at
`validation/storyline-gate-structured-20260928`, with manifest
`/home/work/gui-memory-specialization/data/storyline_gate_structured/canonical.jsonl`.
Its new scorer outputs use `structured-formal/<backend>` under that source,
not the copied pilot `formal/` directory.
All three raw-response scorers completed 3,888/3,888 with zero errors, and
their exploratory producer-by-consumer matrix is stored in
`validation/storyline-gate-20260928/selectors/producer_consumer_3x3.json`.
Its matched-versus-transferred episode-bootstrap intervals cross zero for all
three consumers. Because prior replies include free-form reasoning, this is
not the strict gate. The structured LLaVA-NeXT scorer has completed 3,498/3,498
with zero errors, and the structured Qwen2.5-VL and InternVL2 scorers have
started on the same 3,498-row manifest. The structured 3×3 remains pending.
The two manifests must never be mixed in a cache or matrix.
LLaVA-NeXT runs on port 30349, Qwen2.5-VL on 30522, InternVL2 on 30780; each
has eight independent one-GPU worker processes. The new ports were reserved
from `/share/platform/available_port.txt` with an audit in
`validation/storyline-gate-20260928/reserved_ports.json`.
