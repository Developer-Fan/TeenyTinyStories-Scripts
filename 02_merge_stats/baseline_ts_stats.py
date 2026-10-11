"""Part B/C: TinyStories V1 baseline tokenization (TS train + valid).

Read-only. Writes results/baseline_ts.json and results/ts_vocab.json.
"""

import hashlib
import json
import statistics
import time
from array import array
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TOK_ID = "EleutherAI/gpt-neo-125M"
DATASET = "roneneldan/TinyStories"

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize_file(tok, path, cap=400):
    text = Path(path).read_text(encoding="utf-8")
    stories = [b.strip() for b in text.split("<|endoftext|>") if b.strip()]
    eos = tok.eos_token_id
    freq = Counter()
    seen = set()
    lengths_all = array("i")
    lengths_capped = array("i")
    tokens_all = 0
    tokens_capped = 0
    n_capped = 0
    over = 0
    for i, s in enumerate(stories):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        n = len(ids)
        lengths_all.append(n)
        tokens_all += n + 1
        freq.update(ids)
        freq[eos] += 1
        seen.update(ids)
        seen.add(eos)
        if n <= cap:
            n_capped += 1
            tokens_capped += n + 1
            lengths_capped.append(n)
        else:
            over += 1
        if (i + 1) % 500000 == 0:
            log(f"{Path(path).name}: {i + 1:,} stories | {tokens_all:,} tokens")
    sl = sorted(lengths_capped)
    return {
        "path": str(path), "bytes": Path(path).stat().st_size, "sha256": sha256(path),
        "stories": len(stories), "tokens_with_eos": tokens_all,
        "stories_capped": n_capped, "tokens_with_eos_capped": tokens_capped,
        "stories_over_400": over,
        "length_mean_all": round(statistics.mean(lengths_all), 3),
        "length_mean_capped": round(statistics.mean(lengths_capped), 3),
        "length_capped_p5": sl[int(0.05 * (n_capped - 1))],
        "length_capped_p50": sl[int(0.50 * (n_capped - 1))],
        "length_capped_p95": sl[int(0.95 * (n_capped - 1))],
        "tokens_share_over_400": round((tokens_all - tokens_capped) / tokens_all, 6),
    }, freq, seen, lengths_capped


def main():
    api = HfApi()
    info = api.dataset_info(DATASET)
    license_ = None
    try:
        license_ = info.cardData.get("license")
    except Exception:
        pass
    log(f"dataset revision {info.sha} | license {license_}")

    tok = AutoTokenizer.from_pretrained(TOK_ID, clean_up_tokenization_spaces=False)
    assert tok.eos_token_id == 50256, tok.eos_token_id

    train = hf_hub_download(DATASET, "TinyStories-train.txt", repo_type="dataset")
    valid = hf_hub_download(DATASET, "TinyStories-valid.txt", repo_type="dataset")

    train_stats, freq, seen, ts_capped_lengths = tokenize_file(tok, train)
    log(f"train done: {train_stats['stories']:,} stories | {train_stats['tokens_with_eos']:,} tokens | capped {train_stats['tokens_with_eos_capped']:,}")
    valid_stats, _, _, _ = tokenize_file(tok, valid)
    log(f"valid done: {valid_stats['stories']:,} stories | {valid_stats['tokens_with_eos']:,} tokens")

    top10k = [i for i, _ in freq.most_common(10000)]
    tokens_outside_top10k = sum(freq[i] for i in freq if i not in set(top10k))

    out = {
        "dataset": DATASET, "revision": info.sha, "license": license_,
        "tokenizer": TOK_ID, "eos_token_id": tok.eos_token_id,
        "train": train_stats, "valid": valid_stats,
        "vocab": {
            "distinct_ids": len(seen),
            "top10k_tokens_share_outside": round(tokens_outside_top10k / train_stats["tokens_with_eos"], 6),
            "tokens_outside_top10k": tokens_outside_top10k,
        },
    }
    (RESULTS / "baseline_ts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RESULTS / "ts_vocab.json").write_text(json.dumps({
        "seen_ids": sorted(seen),
        "top10k_ids": top10k,
        "freq": {str(k): v for k, v in freq.items()},
    }), encoding="utf-8")
    import numpy as np
    np.save(RESULTS / "ts_capped_lengths.npy", np.asarray(ts_capped_lengths, dtype=np.int32))
    log("wrote results/baseline_ts.json, results/ts_vocab.json, results/ts_capped_lengths.npy")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
