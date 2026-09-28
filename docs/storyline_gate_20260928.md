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
not the strict gate. All three structured scorers completed 3,498/3,498 with
zero errors and 8/8 completion markers each. Each cache passed the same-manifest
SHA, coverage, candidate-order, and finite-loss checks before selector training.
The two manifests must never be mixed in a cache or matrix.
LLaVA-NeXT runs on port 30349, Qwen2.5-VL on 30522, InternVL2 on 30780; each
has eight independent one-GPU worker processes. The new ports were reserved
from `/share/platform/available_port.txt` with an audit in
`validation/storyline-gate-20260928/reserved_ports.json`.

## Structured-action gate result

The three independent selectors fit 2,846 training decisions and used 300
other training decisions for episode-separated calibration. The 352 validation
decisions from 110 held-out episodes were used only for evaluation.

Held-out gold-action cross-entropy per native action token (lower is better).
Rows are the selector/memory producer; columns are frozen consumers. Compare
numbers **within a column**, never absolute losses across different models.

| Producer | Qwen2.5-VL | LLaVA-NeXT | InternVL2 |
|---|---:|---:|---:|
| E_Qwen | **2.1507** | 2.8772 | 2.2576 |
| E_LLaVA | 2.1606 | 2.8765 | 2.2497 |
| E_InternVL | 2.1562 | **2.8684** | **2.2431** |
| Recent-one baseline | 2.1672 | 2.9143 | 2.3196 |

The mean matched-selector advantage over the mean of the two transferred
selectors (positive favors specialization) was Qwen +0.0077, LLaVA -0.0037,
InternVL +0.0106 CE. Episode-bootstrap 95% intervals were respectively
[-0.0083, +0.0231], [-0.0172, +0.0081], and [-0.0073, +0.0304]; **all cross
zero**. Both Qwen and InternVL match their own selector in the point estimate,
but LLaVA does not. This gate therefore does not establish systematic diagonal
specialization. The selectors do improve over recent-one within each consumer,
so the negative specialization result is not the same as saying memory choice
has no effect.

Full machine-readable result:
`validation/storyline-gate-structured-20260928/structured-selectors/producer_consumer_3x3.json`.
This is an offline action-likelihood result with at most three history events,
a one-event output budget, and a compact RGB/action-statistic selector. It does
not measure online task success, failure recovery, or a richer multimodal
history encoder. Do not advance to latent or model-conditioned memory on the
basis of this gate; a stronger canonical history-encoder replication and
wording/field controls would be needed before ruling the hypothesis in or out.
