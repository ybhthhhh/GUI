# Structured-action Storyline gate artifacts

`E_qwen25vl.json`, `E_llava_next.json`, and `E_internvl2.json` are the
independently fitted 15-feature event-selector weights and normalization
statistics for the first canonical cross-model gate. They are **not** VLM
backbone weights or the earlier Q-Former compressors.

`producer_consumer_3x3.json` is the complete 352-decision, 110-episode
held-out evaluation, including producer-by-consumer cross-entropy, fixed
baselines, subgroup results, and episode-bootstrap intervals. Lower CE is
better, but values are comparable only within a consumer column because the
three models have different native tokenizers.

All four files were copied byte-for-byte from
`/home/work/gui-memory-specialization/validation/storyline-gate-structured-20260928/structured-selectors`
after scorer cache verification. The source manifest SHA-256 is
`bb5dcf2b1ce68bc0c28ff8cd9792210e6da1b225bff6de784144a607749a11bb`.
No raw screenshots, trajectories, optimizer states, or pretrained model weights
are included here. The gate is offline action-likelihood only, not executed
GUI success.
