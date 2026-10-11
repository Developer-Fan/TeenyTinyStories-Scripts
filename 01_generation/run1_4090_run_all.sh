#!/bin/bash
# Run 4 shards x 50M tokens in parallel on the Vast instance, then merge.
set -e
cd "$(dirname "$0")"
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "start: $(date)"
pids=()
for i in 0 1 2 3; do
  SHARD_ID=$i MODE=generate nohup python generate.py > "logs/shard$(printf %02d "$i").log" 2>&1 &
  pids+=($!)
  echo "shard $i -> pid ${pids[-1]}"
done
wait "${pids[@]}"
echo "all shards finished: $(date)"
TARGET_TOKENS=0 MODE=merge python generate.py 2>&1 | tee logs/merge.log
echo "merge done: $(date)"
