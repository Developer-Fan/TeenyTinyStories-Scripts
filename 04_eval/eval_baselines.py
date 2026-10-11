"""Evaluate the released roneneldan/TinyStories checkpoints (reference only).

These are pretrained models with an unknown schedule and (per the paper) a top-10K
vocabulary; comparisons with the TTS students are secondary, not matched-pair.

Usage:
  python eval_baselines.py [--models roneneldan/TinyStories-1M roneneldan/TinyStories-3M roneneldan/TinyStories-8M]
"""

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENIZER = "EleutherAI/gpt-neo-125M"
EOS = 50256
SEQ = 512
DEFAULT_MODELS = ["roneneldan/TinyStories-1M", "roneneldan/TinyStories-3M", "roneneldan/TinyStories-8M"]


@torch.no_grad()
def evaluate(model, arr, micro_batch, device, autocast_dtype):
    model.eval()
    n_blocks = (len(arr) - 1) // SEQ
    total_loss, total_tokens = 0.0, 0
    for b in range(0, n_blocks, micro_batch):
        idx = list(range(b, min(b + micro_batch, n_blocks)))
        x = torch.stack([torch.from_numpy(np.asarray(arr[i * SEQ:(i + 1) * SEQ], dtype=np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(np.asarray(arr[i * SEQ + 1:(i + 1) * SEQ + 1], dtype=np.int64)) for i in idx]).to(device)
        ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
        with ctx:
            out = model(input_ids=x, labels=y)
        total_loss += float(out.loss) * x.numel()
        total_tokens += x.numel()
    mean = total_loss / max(total_tokens, 1)
    return {"loss_nats": round(mean, 6), "ppl": round(math.exp(mean), 4), "tokens": total_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.cwd()))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    root = Path(args.root)
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    manifest_path = Path(args.data_manifest) if args.data_manifest else data_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {k: np.memmap(manifest[k]["path"], dtype=np.uint16, mode="r")
              for k in ("tts_val", "ts_val_capped", "ts_val_full")}

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    tok = AutoTokenizer.from_pretrained(TOKENIZER, clean_up_tokenization_spaces=False)
    assert tok.eos_token_id == EOS

    out = {"tokenizer": TOKENIZER, "confounds": ("released pretrained checkpoints; unknown schedule/epochs; "
                                                 "paper reports a top-10K vocabulary; secondary reference only"),
           "models": {}}
    t0 = time.time()
    for name in args.models:
        model = AutoModelForCausalLM.from_pretrained(name)
        assert model.config.vocab_size == 50257, model.config.vocab_size
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        entry = {"params_total": n_params, "hidden_size": model.config.hidden_size,
                 "num_layers": model.config.num_layers, "evals": {}}
        for key, arr in arrays.items():
            entry["evals"][key] = evaluate(model, arr, args.micro_batch, device, autocast_dtype)
            print(f"[{time.time() - t0:7.1f}s] {name} {key}: {entry['evals'][key]}", flush=True)
        out["models"][name] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out["measured"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path = Path(args.out) if args.out else root / "baselines_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
