#!/bin/bash
# One-time pod setup: dependencies + prefetch of model and seed file.
set -e
source /venv/main/bin/activate
uv pip install -q transformers accelerate
python - <<'EOF'
import torch
from huggingface_hub import snapshot_download

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0))
snapshot_download("roneneldan/TinyStories-Instruct-33M")
snapshot_download("roneneldan/TinyStoriesInstruct", repo_type="dataset",
                  allow_patterns=["TinyStories-Instruct-valid.txt"])
print("prefetched model + validation file")
EOF
