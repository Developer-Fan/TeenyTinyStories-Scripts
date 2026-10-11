"""Manual blind rating: shows two completions as One/Two (models hidden, order
randomized per item with a fixed seed), you type 1, 2 or tie. Usage:
  python manual_rate.py [--n 20] [--seed 99] [--pair ours_33M ts_33M]
Saves manual_ratings.csv (model identities recorded, never shown).
"""
import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"


def load_gens():
    rows = {}
    with open(OUT / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["model"], r["prompt_id"], r["completion_idx"])] = r
    return rows


from prompts_lib import load_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--pair", nargs=2, default=None,
                    help="two models, e.g. --pair ours_33M ts_33M (default: all pairs)")
    args = ap.parse_args()
    gens, prompts = load_gens(), load_prompts()
    models = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
    pairs = [tuple(args.pair)] if args.pair else list(itertools.combinations(models, 2))
    rng = random.Random(args.seed)
    items = []
    for _ in range(args.n):
        a, b = rng.choice(pairs)
        items.append((rng.randint(1, 50), a, b))
    outp = OUT / "manual_ratings.csv"
    new = not outp.exists()
    fout = open(outp, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(["model_one", "model_two", "prompt_id", "shown_first",
                    "shown_second", "verdict", "winner"])
    print(f"{len(items)} items. Type 1, 2 or tie (Ctrl-C to stop; progress is saved).")
    try:
        for i, (pid, a, b) in enumerate(items):
            ta, tb = gens[(a, pid, 0)]["text"], gens[(b, pid, 0)]["text"]
            flip = rng.random() < 0.5
            first, second = (tb, ta) if flip else (ta, tb)
            print(f"\n--- {i+1}/{len(items)} (prompt {pid}) ---\nBEGINNING: {prompts[pid]}")
            print(f"\n[ONE]\n{first}\n\n[TWO]\n{second}")
            while True:
                v = input("better? 1 / 2 / tie: ").strip().lower()
                if v in ("1", "2", "tie", "t", "0"):
                    break
                print("type 1, 2 or tie")
            v = {"t": "tie", "0": "tie"}.get(v, v)
            winner = "tie" if v == "tie" else ((b if flip else a) if v == "1" else (a if flip else b))
            w.writerow([a, b, pid, b if flip else a, a if flip else b, v, winner])
            fout.flush()
            print(f"recorded: {winner}")
    except KeyboardInterrupt:
        print("\nstopped; progress saved.")
    fout.close()


if __name__ == "__main__":
    main()
