"""Aggregate easy GPT-eval: strict parse of turn-2 replies, per-prompt means,
per-model means + bootstrap CIs, paired diffs, age distribution.
Reads analyse/easy_judge_raw/judge_raw_egpteval2.jsonl (+eg2 retry files).
Writes analyse/easy_judge_raw/egpteval_results.json
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RX = re.compile(r"^\s*Grammar:\s*(\d{1,2})\s*/\s*10\s*,\s*Creativity:\s*(\d{1,2})"
                r"\s*/\s*10\s*,\s*Consistency:\s*(\d{1,2})\s*/\s*10\s*,"
                r"\s*Age group:\s*([A-F])\b")


def load_raw(path):
    rows, fails = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            base = r["key"].split("#")[0]
            if base in rows:
                continue
            if r.get("error"):
                fails[base] = fails.get(base, []) + [f"transport:{r['error']}"]
                continue
            m = RX.search((r["response_text"] or "").replace("**", ""))
            if not m:
                fails[base] = fails.get(base, []) + ["parse"]
                continue
            g, c, s, a = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            if not (0 <= g <= 10 and 0 <= c <= 10 and 0 <= s <= 10):
                fails[base] = fails.get(base, []) + ["range"]
                continue
            rows[base] = {"g": g, "c": c, "s": s, "age": a}
            fails.pop(base, None)
    return rows, fails


def main():
    rows, fails = {}, {}
    parts = sorted([str(HERE / "judge_raw_egpteval2.jsonl")] +
                   glob.glob(str(HERE / "judge_raw_egretry*_g2.jsonl")))
    for part in parts:
        try:
            r, f = load_raw(part)
        except FileNotFoundError:
            continue
        for k, v in r.items():
            rows.setdefault(k, v)
        for k, v in f.items():
            if k not in rows:
                fails[k] = v
    for k in list(fails):
        if k in rows:
            del fails[k]
    print(f"parsed={len(rows)} failed={len(fails)}")
    for k, v in sorted(fails.items())[:20]:
        print("FAIL", k, v)
    per = {}
    for base, v in rows.items():
        _, model, pid, ci = base.split("|")
        per.setdefault((model, int(pid)), []).append(v)
    models = sorted(set(m for m, _ in per))
    rng = np.random.default_rng(7)
    out = {}
    for m in models:
        pids = sorted(set(p for (mm, p) in per if mm == m))
        means = {k: [] for k in ("g", "c", "s")}
        ages = []
        for p in pids:
            vs = per[(m, p)]
            for k in means:
                means[k].append(float(np.mean([v[k] for v in vs])))
            ages.extend(v["age"] for v in vs)
        res = {}
        for k, vals in means.items():
            v = np.asarray(vals)
            bs = np.mean(rng.choice(v, size=(10000, len(v)), replace=True), axis=1)
            res[k] = {"mean": float(v.mean()), "lo": float(np.percentile(bs, 2.5)),
                      "hi": float(np.percentile(bs, 97.5))}
        res["age_dist"] = {a: ages.count(a) for a in "ABCDEF"}
        res["n_prompts"] = len(pids)
        out[m] = res
    pairs = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            common = sorted(set(p for (mm, p) in per if mm == a) &
                            set(p for (mm, p) in per if mm == b))
            d = {}
            for k in ("g", "c", "s"):
                va = np.array([np.mean([v[k] for v in per[(a, p)]]) for p in common])
                vb = np.array([np.mean([v[k] for v in per[(b, p)]]) for p in common])
                diff = va - vb
                bs = np.mean(rng.choice(diff, size=(10000, len(diff)), replace=True), axis=1)
                d[k] = {"mean": float(diff.mean()), "lo": float(np.percentile(bs, 2.5)),
                        "hi": float(np.percentile(bs, 97.5))}
            pairs[f"{a}_vs_{b}"] = d
    json.dump({"models": out, "paired": pairs, "n_failed": len(fails),
               "failed_keys": sorted(fails)},
              open(HERE / "egpteval_results.json", "w"), indent=2)
    print("wrote egpteval_results.json")


if __name__ == "__main__":
    main()
