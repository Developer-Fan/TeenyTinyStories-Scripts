"""Part B/C/D: TTS token statistics, validation split, vocab overlap, KS vs capped TS.

Read-only. Writes results/tts_token_stats.json.
"""

import hashlib
import json
import random
import re
import time
from array import array
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TTS_TXT = ROOT / "merged/tts_corpus.txt"
TTS_JSONL = ROOT / "merged/stories_merged.jsonl"
DATA_SEED = 1234
VAL_STORIES = 20000
TOK_ID = "EleutherAI/gpt-neo-125M"
WS = re.compile(r"\s+")

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


def norm(s):
    return WS.sub(" ", s.lower()).strip()


def main():
    ts = json.loads((RESULTS / "baseline_ts.json").read_text(encoding="utf-8"))
    vocab = json.loads((RESULTS / "ts_vocab.json").read_text(encoding="utf-8"))
    seen_ts = set(vocab["seen_ids"])
    top10k = set(vocab["top10k_ids"])

    tok = AutoTokenizer.from_pretrained(TOK_ID, clean_up_tokenization_spaces=False)
    eos = tok.eos_token_id
    assert eos == 50256

    n_stories = 2511490
    rng = random.Random(DATA_SEED)
    order = list(range(n_stories))
    rng.shuffle(order)
    val_idx = set(order[:VAL_STORIES])
    log(f"split defined: DATA_SEED={DATA_SEED}, val stories={len(val_idx)}")

    text = TTS_TXT.read_text(encoding="utf-8")
    stories = [b.strip() for b in text.split("<|endoftext|>") if b.strip()]
    assert len(stories) == n_stories, (len(stories), n_stories)

    distinct_tts = set()
    out_of_support = 0
    outside_top10k = 0
    tokens_all = 0
    val_tokens = 0
    val_stories_text = {}
    train_hashes = set()
    train_first12 = set()
    val_first12_collisions = 0
    lengths_train = array("i")

    for i, s in enumerate(stories):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        n = len(ids)
        tokens_all += n + 1
        distinct_tts.update(ids)
        distinct_tts.add(eos)
        for x in ids:
            if x not in seen_ts:
                out_of_support += 1
            if x not in top10k:
                outside_top10k += 1
        words = re.findall(r"[a-z']+", s.lower())
        f12 = hash(tuple(words[:12]))
        h = hashlib.md5(norm(s).encode()).digest()
        if i in val_idx:
            val_tokens += n + 1
            val_stories_text[i] = (h, f12)
        else:
            lengths_train.append(n)
            train_hashes.add(h)
            train_first12.add(f12)
        if (i + 1) % 500000 == 0:
            log(f"tokenized {i + 1:,} stories | {tokens_all:,} tokens | out-of-support {out_of_support:,}")

    val_exact_dups = sum(1 for h, _ in val_stories_text.values() if h in train_hashes)
    val_f12_collisions = sum(1 for _, f in val_stories_text.values() if f in train_first12)

    ts_capped = np.load(RESULTS / "ts_capped_lengths.npy")
    tts_train_len = np.asarray(lengths_train, dtype=np.int32)
    grid = np.union1d(ts_capped, tts_train_len)
    cdf_ts = np.searchsorted(np.sort(ts_capped), grid, side="right") / len(ts_capped)
    cdf_tts = np.searchsorted(np.sort(tts_train_len), grid, side="right") / len(tts_train_len)
    ks = float(np.max(np.abs(cdf_ts - cdf_tts)))

    total_with_eos = tokens_all
    train_with_eos = total_with_eos - val_tokens
    total_no_eos = total_with_eos - n_stories
    train_no_eos = total_no_eos - (val_tokens - VAL_STORIES)
    ts_capped_with_eos = ts["train"]["tokens_with_eos_capped"]
    gap_with_eos = train_with_eos - ts_capped_with_eos
    gap_no_eos = train_no_eos - (ts_capped_with_eos - ts["train"]["stories_capped"])
    gap_pct = gap_with_eos / ts_capped_with_eos

    out = {
        "split": {"data_seed": DATA_SEED, "val_stories": VAL_STORIES, "val_tokens_with_eos": val_tokens,
                  "train_stories": n_stories - VAL_STORIES,
                  "val_exact_duplicates_vs_train": val_exact_dups,
                  "val_first12_collisions_vs_train": val_f12_collisions},
        "tokens": {"total_with_eos": total_with_eos, "total_no_eos": total_no_eos,
                    "train_with_eos": train_with_eos, "train_no_eos": train_no_eos},
        "vocab": {"distinct_ids_tts": len(distinct_tts),
                   "out_of_support_tokens": out_of_support,
                   "out_of_support_rate": round(out_of_support / (total_with_eos - n_stories), 8),
                   "tokens_outside_top10k": outside_top10k,
                   "outside_top10k_rate": round(outside_top10k / (total_with_eos - n_stories), 8)},
        "ts_capped_train": {"tokens_with_eos": ts_capped_with_eos,
                             "tokens_no_eos": ts_capped_with_eos - ts["train"]["stories_capped"],
                             "stories": ts["train"]["stories_capped"]},
        "matching_gap": {"gap_with_eos": gap_with_eos, "gap_no_eos": gap_no_eos,
                          "gap_percent_of_capped_ts": round(gap_pct, 6),
                          "flag": "LARGE" if gap_pct > 0.10 else "small"},
        "length_ks_capped_ts_vs_tts_train": {"ks_statistic": ks,
                                              "ts_mean": ts["train"]["length_mean_capped"],
                                              "tts_train_mean": round(float(tts_train_len.mean()), 3),
                                              "ts_p50": ts["train"]["length_capped_p50"],
                                              "tts_train_p50": int(np.median(tts_train_len)),
                                              "ts_p95": ts["train"]["length_capped_p95"],
                                              "tts_train_p95": int(np.percentile(tts_train_len, 95))},
    }
    (RESULTS / "tts_token_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("wrote results/tts_token_stats.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
