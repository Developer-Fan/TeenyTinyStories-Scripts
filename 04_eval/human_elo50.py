"""Human Elo-50: 25 prompts x 1M pair (ours_1M vs ts_1M) + 25 prompts x 33M pair
(ours_33M vs ts_33M), blinded One/Two, fixed seed. Usage: python human_elo50.py
Saves eval_outputs/human50_ratings.csv. Ctrl-C stops; progress is kept."""
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
    ap.add_argument("--n-per-pair", type=int, default=25)
    ap.add_argument("--seed", type=int, default=50)
    args = ap.parse_args()
    gens, prompts = load_gens(), load_prompts()
    rng = random.Random(args.seed)
    pairs = [("ours_1M", "ts_1M"), ("ours_33M", "ts_33M")]
    used = set()
    items = []
    for a, b in pairs:
        remaining = [p for p in range(1, 51) if p not in used]
        pids = rng.sample(remaining, args.n_per_pair)
        used.update(pids)
        items += [(p, a, b) for p in pids]
    rng.shuffle(items)
    outp = OUT / "human50_ratings.csv"
    new = not outp.exists()
    done = set()
    if not new:
        with open(outp) as f:
            for r in csv.DictReader(f):
                done.add((int(r["prompt_id"]), r["model_a"], r["model_b"]))
    items = [it for it in items if (it[0], it[1], it[2]) not in done]
    fout = open(outp, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(["prompt_id", "model_a", "model_b", "shown_first",
                    "shown_second", "verdict", "winner"])
    print(f"{len(items)} items (1M pair + 33M pair), blinded. Type 1, 2 or tie.")
    try:
        for i, (pid, a, b) in enumerate(items):
            ta, tb = gens[(a, pid, 0)]["text"], gens[(b, pid, 0)]["text"]
            flip = rng.random() < 0.5
            first, second = (tb, ta) if flip else (ta, tb)
            print(f"\n--- {i+1}/{len(items)} (prompt {pid}) ---\nBEGINNING: {prompts[pid]}")
            print(f"\n[ONE]\n{first}\n\n[TWO]\n{second}")
            while True:
                v = input("better? 1 / 2 / tie: ").strip().lower()
                if v in ("1", "2", "tie", "t"):
                    break
                print("type 1, 2 or tie")
            v = "tie" if v in ("tie", "t") else v
            first_is, second_is = (b, a) if flip else (a, b)
            winner = "tie" if v == "tie" else (first_is if v == "1" else second_is)
            w.writerow([pid, a, b, first_is, second_is, v, winner])
            fout.flush()
            print(f"recorded: {winner}")
    except KeyboardInterrupt:
        print("\nstopped; progress saved.")
    fout.close()


if __name__ == "__main__":
    main()
