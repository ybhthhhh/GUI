#!/usr/bin/env bash
# A real-data, single-domain Stage-1a run used while the larger multi-domain
# CoMEM corpus is still being staged.  It is deliberately separate from the
# later shopping+service main experiment and preserves its checkpoint lineage.
set -euo pipefail

WORK=/root/gui-memory-specialization
PYTHON=/root/.venvs/comem-train/bin/python
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:$WORK/src
export LD_LIBRARY_PATH=/usr/local/corex/lib64

raw=$WORK/data/comem_raw
stage=$WORK/data/stage1
logs=$WORK/logs
mkdir -p "$raw" "$stage" "$logs"

while [[ ! -f "$raw/services.zip" ]]; do
  if ! pgrep -f "[p]ython3 .*download_comem_subset.py.*services.zip" >/dev/null; then
    echo "STAGE1A_ERROR missing_or_failed_shard=services.zip" >&2
    exit 1
  fi
  sleep 20
done

"$PYTHON" "$WORK/examples/prepare_comem_stage1.py" \
  "$raw/services.zip" "$stage/services_success.jsonl" \
  --image-root "$stage/images" --max-samples 15000 \
  >"$logs/prepare-services-stage1a.log" 2>&1

# A bounded, genuine four-GPU run.  The later main experiment uses the
# larger composite corpus and a separate output directory.
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$WORK/examples/train_memw_stage1.py" "$stage/services_success.jsonl" \
  "$WORK/outputs/stage1a-services" --model-path /home/work/Qwen2.5-VL-7B-Instruct \
  --max-steps 500 --checkpoint-every 50 --distributed \
  >"$logs/stage1a-services.log" 2>&1

CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$WORK/examples/evaluate_memw_stage1.py" \
  "$stage/services_success.jsonl" "$WORK/outputs/stage1a-services/stage1_last.pt" \
  "$WORK/outputs/stage1a-services/held_out.json" --model-path /home/work/Qwen2.5-VL-7B-Instruct \
  --max-samples 32 >"$logs/stage1a-services-held-out.log" 2>&1
