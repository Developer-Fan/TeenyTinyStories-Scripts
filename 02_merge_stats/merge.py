import argparse
import glob
import hashlib
import json
import re
import time
from pathlib import Path

NORM = re.compile(r"\s+")


def norm(s):
    return NORM.sub(" ", s.lower()).strip()


def parse_args():
    p = argparse.ArgumentParser(description="Merge TeenyTinyStories shards from one or more directories.")
    p.add_argument("--dirs", nargs="+",
                   default=["vast/Corpus_TeenyTinyStories", "Corpus_TeenyTinyStories", "vast2/Corpus_TeenyTinyStories"],
                   help="Directories searched for stories_shard*.jsonl (order matters; first occurrence wins)")
    p.add_argument("--out-dir", default="merged")
    p.add_argument("--max-tokens", type=int, default=0, help="Stop after this many accepted tokens (0 = no cap)")
    p.add_argument("--log-every", type=int, default=250000)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "stories_merged.jsonl"
    out_txt = out_dir / "tts_corpus.txt"
    out_summary = out_dir / "merge_summary.json"

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    files = []
    for d in args.dirs:
        found = sorted(glob.glob(str(Path(d) / "stories_shard*.jsonl")))
        if not found:
            log(f"warning: no stories_shard*.jsonl in {d}")
        files.extend(found)
    files = [f for f in files if Path(f).name != "stories_merged.jsonl"]
    if not files:
        raise SystemExit("no shard files found")
    log(f"merging {len(files)} files from {len(args.dirs)} dirs")
    for f in files:
        log(f"  {f}")

    seen = set()
    n_in = n_out = dups = tokens = 0
    all3_ok = all3_n = 0
    per_file = {}

    with out_jsonl.open("w", encoding="utf-8") as fj, out_txt.open("w", encoding="utf-8") as ft:
        for fp in files:
            cnt = ftok = 0
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    n_in += 1
                    r = json.loads(line)
                    story = r.get("story", "")
                    key = hashlib.md5(norm(story).encode()).digest()
                    if key in seen:
                        dups += 1
                        continue
                    seen.add(key)
                    r["id"] = n_out
                    fj.write(json.dumps(r) + "\n")
                    ft.write(story + "\n<|endoftext|>\n")
                    n_out += 1
                    t = int(r.get("n_tokens") or 0)
                    tokens += t
                    cnt += 1
                    ftok += t
                    hw = r.get("has_words")
                    if hw:
                        all3_n += 1
                        if all(hw):
                            all3_ok += 1
                    if n_out % args.log_every == 0:
                        log(f"{n_out:,} stories | {tokens:,} tokens | dups {dups:,} | {n_out / (time.time() - t0):.0f} stories/s")
                    if args.max_tokens and tokens >= args.max_tokens:
                        break
            per_file[fp] = {"stories": cnt, "tokens": ftok}
            log(f"done {fp}: {cnt:,} stories | {ftok:,} tokens")
            if args.max_tokens and tokens >= args.max_tokens:
                break

    summary = {
        "inputs": files,
        "stories_in": n_in,
        "stories_out": n_out,
        "duplicates_dropped": dups,
        "tokens_out": tokens,
        "mean_tokens": round(tokens / max(n_out, 1), 2),
        "seed_all3_rate": round(all3_ok / max(all3_n, 1), 4),
        "seed_all3_n": all3_n,
        "per_file": per_file,
        "out_jsonl": str(out_jsonl),
        "out_txt": str(out_txt),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"merged {n_out:,} stories | {tokens:,} tokens | {dups:,} duplicates dropped")
    log(f"wrote {out_jsonl} and {out_txt}")
    log(f"summary: {out_summary}")


if __name__ == "__main__":
    main()
