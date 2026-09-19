#!/usr/bin/env bash
# Run entirely on the training node under nohup.  Every large transfer is
# resumable; a failed download is terminal rather than silently producing a
# partial corpus.
set -euo pipefail

WORK=/root/gui-memory-specialization
PYTHON=/root/.venvs/comem-train/bin/python
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:$WORK/src
export LD_LIBRARY_PATH=/usr/local/corex/lib64

raw=$WORK/data/comem_raw
stage=$WORK/data/stage1
logs=$WORK/logs
mkdir -p "$raw" "$stage" "$logs"

wait_for_shard() {
  local target=$1
  local name=$2
  while [[ ! -f "$target" ]]; do
    if ! pgrep -f "[p]ython3 .*download_comem_subset.py.*${name}" >/dev/null; then
      echo "PIPELINE_ERROR missing_or_failed_shard=$name" >&2
      exit 1
    fi
    sleep 20
  done
}

# shopping download is launched independently so this coordinator can be
# restarted without duplicating the transfer.
wait_for_shard "$raw/shopping.zip" shopping.zip
"$PYTHON" "$WORK/examples/prepare_comem_stage1.py" \
  "$raw/shopping.zip" "$stage/shopping_success.jsonl" \
  --image-root "$stage/images" --max-samples 15000 \
  >"$logs/prepare-shopping.log" 2>&1

if [[ ! -f "$raw/service.zip" ]]; then
  "$PYTHON" "$WORK/examples/download_comem_subset.py" \
    expand_memory/expand_memory_part1/service.zip "$raw/service.zip" \
    --expected-bytes 18902536174 --proxy http://127.0.0.1:7897 \
    >"$logs/download-service.log" 2>&1
fi
"$PYTHON" "$WORK/examples/prepare_comem_stage1.py" \
  "$raw/service.zip" "$stage/service_success.jsonl" \
  --image-root "$stage/images" --max-samples 15000 \
  >"$logs/prepare-service.log" 2>&1
"$PYTHON" "$WORK/examples/merge_stage1_manifests.py" \
  "$stage/comem_shopping_service.jsonl" "$stage/shopping_success.jsonl" "$stage/service_success.jsonl" \
  >"$logs/merge-stage1.log" 2>&1

# Prove the same synchronous communication path used for the formal launch.
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$WORK/examples/train_memw_stage1.py" "$stage/comem_shopping_service.jsonl" \
  "$WORK/outputs/stage1-ddp-smoke" --model-path /home/work/Qwen2.5-VL-7B-Instruct \
  --max-steps 1 --checkpoint-every 1 --distributed \
  >"$logs/stage1-ddp-smoke.log" 2>&1

# Four GPUs train one synchronized Q-Former; Qwen remains frozen on every rank.
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$WORK/examples/train_memw_stage1.py" "$stage/comem_shopping_service.jsonl" \
  "$WORK/outputs/stage1-formal" --model-path /home/work/Qwen2.5-VL-7B-Instruct \
  --max-steps 5000 --checkpoint-every 50 --distributed \
  >"$logs/stage1-formal.log" 2>&1

# Evaluation is single-process and read-only: it reports the same CE/KL terms
# on episodes excluded from all training updates.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$WORK/examples/evaluate_memw_stage1.py" \
  "$stage/comem_shopping_service.jsonl" "$WORK/outputs/stage1-formal/stage1_last.pt" \
  "$WORK/outputs/stage1-formal/held_out.json" --model-path /home/work/Qwen2.5-VL-7B-Instruct \
  --max-samples 32 >"$logs/stage1-held-out.log" 2>&1
