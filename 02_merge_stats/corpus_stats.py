"""Part A: TeenyTinyStories corpus statistics.

Read-only. Writes results/corpus_stats.json and results/corpus_stats.md.
Does not tokenize the full corpus (see baseline_token_stats.py for Part B/C).
"""

import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from array import array
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
MERGED_JSONL = ROOT / "merged/stories_merged.jsonl"
MERGED_TXT = ROOT / "merged/tts_corpus.txt"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

WORD = re.compile(r"[a-z']+")
WS = re.compile(r"\s+")
HDR = re.compile(r"(^|\n)\s*(Summary|Features|Words|Random sentence|Story):")
TERM = re.compile(r"[.!?][\"'\u201d\u2019)]*\s*$")
SEED_PRESENCE_SEED = 0
DISTINCT_SEED = 1
MEMO_SEED = 2
SHUFFLE_CONTROL_N = 100000
DISTINCT_WORDS = 800000
MEMO_N = 2000

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


def norm(s):
    return WS.sub(" ", s.lower()).strip()


def lenient_match(text, w):
    stem = w[:-1] if (w.endswith("e") and len(w) > 3) else w
    return re.search(r"\b" + re.escape(stem), text.lower()) is not None


def distinct_n_from_words(words, n):
    total = max(0, len(words) - n + 1)
    grams = set(tuple(words[i:i + n]) for i in range(total))
    return (len(grams) / total if total else 0.0), total


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    n_lines = 0
    with open(MERGED_JSONL, "rb") as f:
        for _ in f:
            n_lines += 1
    log(f"TTS merged stories: {n_lines:,}")

    rng = random.Random(DISTINCT_SEED)
    distinct_idx = set(rng.sample(range(n_lines), min(12000, n_lines)))
    rng2 = random.Random(MEMO_SEED)
    memo_idx = set(rng2.sample(range(n_lines), min(MEMO_N, n_lines)))
    rng3 = random.Random(SEED_PRESENCE_SEED)
    control_reservoir = []
    control_seen = 0

    lengths = array("i")
    tokens = 0
    hashes = set()
    dup_stories = 0
    leak = 0
    terminal = 0
    first12 = set()
    first12_dups = 0
    seed_pos = [0, 0, 0]
    seed_all3 = 0
    seed_n = 0
    feat_sizes = Counter()
    per_shard = {}
    distinct_words = []
    distinct_word_count = 0
    memo_stories = {}
    shard_sample_words = {0: [], 9: []}
    shard_sample_counts = {0: 0, 9: 0}

    with open(MERGED_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            story = r["story"]
            nt = int(r.get("n_tokens") or 0)
            lengths.append(nt)
            tokens += nt

            h = hashlib.md5(norm(story).encode()).digest()
            if h in hashes:
                dup_stories += 1
            else:
                hashes.add(h)

            if HDR.search(story):
                leak += 1
            if TERM.search(story):
                terminal += 1

            words = WORD.findall(story.lower())
            key = tuple(words[:12])
            kh = hash(key)
            if kh in first12:
                first12_dups += 1
            else:
                first12.add(kh)

            hw = r.get("has_words")
            if hw:
                seed_n += 1
                for j in range(3):
                    if hw[j]:
                        seed_pos[j] += 1
                if all(hw):
                    seed_all3 += 1
            feat_sizes[len(r.get("seed", {}).get("features", []))] += 1

            sh = int(r.get("shard", -1))
            d = per_shard.setdefault(sh, {"stories": 0, "tokens": 0, "cap400": 0, "leak": 0, "terminal": 0,
                                          "seed_pos": [0, 0, 0], "seed_all3": 0, "seed_n": 0})
            d["stories"] += 1
            d["tokens"] += nt
            if nt >= 400:
                d["cap400"] += 1
            if HDR.search(story):
                d["leak"] += 1
            if TERM.search(story):
                d["terminal"] += 1
            if hw:
                d["seed_n"] += 1
                for j in range(3):
                    if hw[j]:
                        d["seed_pos"][j] += 1
                if all(hw):
                    d["seed_all3"] += 1

            if i in distinct_idx and distinct_word_count < DISTINCT_WORDS:
                distinct_words.extend(words)
                distinct_word_count += len(words)
            if i in memo_idx:
                memo_stories[i] = (story, words)
            if sh in shard_sample_words and shard_sample_counts[sh] < 4000:
                shard_sample_words[sh].extend(words)
                shard_sample_counts[sh] += 1

            control_seen += 1
            if len(control_reservoir) < SHUFFLE_CONTROL_N:
                control_reservoir.append((story, list(r.get("seed", {}).get("words", []))))
            else:
                j = rng3.randrange(control_seen)
                if j < SHUFFLE_CONTROL_N:
                    control_reservoir[j] = (story, list(r.get("seed", {}).get("words", [])))

            if (i + 1) % 500000 == 0:
                log(f"scanned {i + 1:,} stories | {tokens:,} tokens")

    log("scan done; computing derived stats")

    lengths_np = lengths
    n = len(lengths_np)
    sorted_len = sorted(lengths_np)
    q = lambda p: sorted_len[min(n - 1, int(p * (n - 1)))]
    bins = list(range(0, 401, 25)) + [10 ** 9]
    hist = Counter()
    for L in lengths_np:
        for b in range(len(bins) - 1):
            if bins[b] <= L < bins[b + 1]:
                hist[f"{bins[b]}-{'400+' if bins[b + 1] == 10 ** 9 else bins[b + 1]}"] += 1
                break

    shuffled = [w for _, w in control_reservoir]
    rng4 = random.Random(SEED_PRESENCE_SEED)
    rng4.shuffle(shuffled)
    ctrl_pos = [0, 0, 0]
    ctrl_n = 0
    for (story, _), words in zip(control_reservoir, shuffled):
        if len(words) != 3:
            continue
        ctrl_n += 1
        for j in range(3):
            if lenient_match(story, words[j]):
                ctrl_pos[j] += 1

    ts_valid = hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt", repo_type="dataset")
    ts_valid_stories = [b.strip() for b in Path(ts_valid).read_text(encoding="utf-8").split("<|endoftext|>") if b.strip()]
    ts_words = []
    for s in ts_valid_stories:
        ts_words.extend(WORD.findall(s.lower()))
        if len(ts_words) >= DISTINCT_WORDS:
            break
    ts_words = ts_words[:DISTINCT_WORDS]
    d_ts = {k: distinct_n_from_words(ts_words, k) for k in (1, 2, 3)}

    d_tts = {k: distinct_n_from_words(distinct_words[:DISTINCT_WORDS], k) for k in (1, 2, 3)}
    d_shard = {}
    for sh, ws in shard_sample_words.items():
        d_shard[sh] = {k: distinct_n_from_words(ws, k) for k in (1, 2, 3)}

    ts13 = set()
    for s in ts_valid_stories:
        w = WORD.findall(s.lower())
        for i in range(len(w) - 12):
            ts13.add(hash(tuple(w[i:i + 13])))
    ts_valid_norm = set(norm(s) for s in ts_valid_stories)
    memo_exact = 0
    memo_13_hits = 0
    memo_n = 0
    for story, words in memo_stories.values():
        memo_n += 1
        if norm(story) in ts_valid_norm:
            memo_exact += 1
        g = set(hash(tuple(words[i:i + 13])) for i in range(max(0, len(words) - 12)))
        if g & ts13:
            memo_13_hits += 1

    as_generated = {}
    for d in [ROOT / "vast/run1_4090/shard_meta", ROOT / "vast/run2_4090/shard_meta",
              ROOT / "vast/mac_04_05/shard_meta", ROOT / "archive"]:
        for f in sorted(d.glob("summary_shard*.json")):
            s = json.loads(f.read_text(encoding="utf-8"))
            as_generated[str(f.relative_to(ROOT))] = {
                "stories": s.get("stories"),
                "tokens": s.get("story_tokens"),
                "session_rejections": s.get("session_rejections"),
            }

    out = {
        "inputs": {
            "merged_jsonl": {"path": str(MERGED_JSONL), "bytes": MERGED_JSONL.stat().st_size, "sha256": sha256(MERGED_JSONL)},
            "merged_txt": {"path": str(MERGED_TXT), "bytes": MERGED_TXT.stat().st_size, "sha256": sha256(MERGED_TXT)},
            "ts_valid": {"path": ts_valid, "bytes": Path(ts_valid).stat().st_size, "sha256": sha256(ts_valid)},
        },
        "merged": {
            "stories": n,
            "tokens_no_eos": tokens,
            "tokens_with_eos": tokens + n,
            "length": {"mean": round(statistics.mean(lengths_np), 3), "median": q(0.5), "p5": q(0.05),
                        "p95": q(0.95), "min": min(lengths_np), "max": max(lengths_np)},
            "histogram": dict(hist),
            "share_at_cap_400": round(sum(1 for L in lengths_np if L >= 400) / n, 6),
            "exact_duplicates": dup_stories,
            "header_leakage": leak,
            "terminal_punctuation_rate": round(terminal / n, 6),
            "first12_collision_stories": first12_dups,
            "seed_presence": {"n": seed_n, "per_position": [round(x / seed_n, 6) for x in seed_pos],
                               "all_three": round(seed_all3 / seed_n, 6)},
            "shuffled_seed_control": {"n": ctrl_n, "per_position": [round(x / ctrl_n, 6) for x in ctrl_pos]},
            "feature_set_size_distribution": {str(k): v for k, v in sorted(feat_sizes.items())},
            "distinct_1_2_3": {"tts": {str(k): round(v[0], 6) for k, v in d_tts.items()},
                                "ts_valid_slice": {str(k): round(v[0], 6) for k, v in d_ts.items()},
                                "words_compared": min(len(distinct_words), DISTINCT_WORDS)},
            "first_vs_last_shard_distinct_1_2_3": {str(k): {str(nn): round(vv[0], 6) for nn, vv in v.items()} for k, v in d_shard.items()},
            "first_vs_last_shard_words": {str(k): len(v) for k, v in shard_sample_words.items()},
            "near_duplicate": {"source": "analysis/minhash_results_full.json", "candidate_pairs": 0, "verified_pairs": 0,
                                "note": "word 5-gram shingles, 128 perms, 16x8 bands, Jaccard>=0.8; see file"},
            "memorization": {"n_stories": memo_n, "reference": "TinyStories-valid (21,990 stories)",
                              "exact_matches": memo_exact, "stories_with_any_13gram_overlap": memo_13_hits},
        },
        "per_shard": {str(k): v for k, v in sorted(per_shard.items())},
        "as_generated_vs_postfilter": as_generated,
    }
    (RESULTS / "corpus_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    m = out["merged"]
    md = ["# TTS corpus statistics (Part A)", "",
          f"- Input: `{MERGED_JSONL}` ({MERGED_JSONL.stat().st_size:,} bytes), sha256 `{out['inputs']['merged_jsonl']['sha256'][:16]}...`",
          f"- Stories: {m['stories']:,} | tokens (no EOS): {m['tokens_no_eos']:,} | tokens (with EOS): {m['tokens_with_eos']:,}",
          f"- Length mean/median/p5/p95/min/max: {m['length']['mean']} / {m['length']['median']} / {m['length']['p5']} / {m['length']['p95']} / {m['length']['min']} / {m['length']['max']}",
          f"- Share at 400-token cap: {m['share_at_cap_400']:.4%}",
          f"- Exact duplicates: {m['exact_duplicates']:,} | header leakage: {m['header_leakage']:,} | terminal punctuation: {m['terminal_punctuation_rate']:.4%}",
          f"- First-12-word collisions: {m['first12_collision_stories']:,}",
          f"- Seed presence (verb/noun/adj, all three): {m['seed_presence']['per_position']} / {m['seed_presence']['all_three']} (n={m['seed_presence']['n']:,})",
          f"- Shuffled control: {m['shuffled_seed_control']['per_position']} (n={m['shuffled_seed_control']['n']:,})",
          f"- Distinct-1/2/3 TTS vs TS-valid slice ({m['distinct_1_2_3']['words_compared']:,} words): {m['distinct_1_2_3']['tts']} vs {m['distinct_1_2_3']['ts_valid_slice']}",
          f"- Memorization ({m['memorization']['n_stories']:,} stories vs TS valid): exact matches {m['memorization']['exact_matches']}, any 13-gram overlap {m['memorization']['stories_with_any_13gram_overlap']}",
          "", "## Histogram", ""]
    md += [f"- {k}: {v:,}" for k, v in m["histogram"].items()]
    (RESULTS / "corpus_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("wrote results/corpus_stats.json and .md")
    print(json.dumps({"stories": n, "tokens_no_eos": tokens, "exact_dups": dup_stories, "leak": leak}, indent=2))


if __name__ == "__main__":
    main()
