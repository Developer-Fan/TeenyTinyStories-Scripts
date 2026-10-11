"""Human eval, starter set: 10 prompts, 33M pair only (ours_33M vs ts_33M).

Blinded (One/Two, order randomized with fixed seed 33), models never shown.
Usage: python human_33m.py [--n 10] [--seed 33]
Saves eval_outputs/human33m_ratings.csv. Ctrl-C stops; progress is kept.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"
sys.path.insert(0, str(HERE))
from prompts_lib import load_prompts


def load_gens():
    import json
    rows = {}
    with open(OUT / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["model"], r["prompt_id"], r["completion_idx"])] = r
    assert len(rows) == 2000, f"frozen set incomplete: {len(rows)}"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()
    gens, prompts = load_gens(), load_prompts()
    rng = random.Random(args.seed)
    pids = rng.sample(range(1, 51), args.n)
    outp = OUT / "human33m_ratings.csv"
    new = not outp.exists()
    fout = open(outp, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(["prompt_id", "shown_first", "shown_second",
                    "verdict", "winner"])
    print(f"{args.n} items, ours_33M vs ts_33M, blinded. Type 1, 2 or tie.")
    try:
        for i, pid in enumerate(pids):
            ta, tb = gens[("ours_33M", pid, 0)]["text"], gens[("ts_33M", pid, 0)]["text"]
            flip = rng.random() < 0.5
            first, second = (tb, ta) if flip else (ta, tb)
            print(f"\n--- {i+1}/{len(pids)} (prompt {pid}) ---\nBEGINNING: {prompts[pid]}")
            print(f"\n[ONE]\n{first}\n\n[TWO]\n{second}")
            while True:
                v = input("better? 1 / 2 / tie: ").strip().lower()
                if v in ("1", "2", "tie", "t"):
                    break
                print("type 1, 2 or tie")
            v = "tie" if v in ("tie", "t") else v
            first_is, second_is = ("ts_33M", "ours_33M") if flip else ("ours_33M", "ts_33M")
            winner = "tie" if v == "tie" else (first_is if v == "1" else second_is)
            w.writerow([pid, first_is, second_is, v, winner])
            fout.flush()
            print(f"recorded: {winner}")
    except KeyboardInterrupt:
        print("\nstopped; progress saved.")
    fout.close()


if __name__ == "__main__":
    main()
