#!/bin/bash
# Run 4 shards (06-09) x 50M tokens in parallel on the Vast instance.
# No merge here: all shards are downloaded and merged locally on the Mac.
set -e
cd "$(dirname "$0")"
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "start: $(date)"
pids=()
for i in 6 7 8 9; do
  SHARD_ID=$i MODE=generate nohup python generate.py > "logs/shard$(printf %02d "$i").log" 2>&1 &
  pids+=($!)
  echo "shard $i -> pid ${pids[-1]}"
done
wait "${pids[@]}"
echo "all shards finished: $(date)"
