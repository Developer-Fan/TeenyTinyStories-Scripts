"""Lenient sensitivity aggregate: recover failed keys whose g/c/s/age scores are
present but separated by a judge-improvised extra field (Plot Logic / Plot /
Plot coherence / Plot makes sense). N/A refusals (no scores) stay missing.
Writes analyse/easy_judge_raw/egpteval_results_lenient.json
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LENIENT = re.compile(r"Grammar:\s*\**(\d{1,2})\s*/\s*10\**,?\s*Creativity:\s*\**(\d{1,2})\s*/\s*10\**,?\s*"
                     r"Consistency:\s*\**(\d{1,2})\s*/\s*10\**,?\s*"
                     r"(?:[^,]+?:\s*\**\d{1,2}\s*/\s*10\**,?\s*)?"
                     r"Age group:\s*\**([A-F])\b")


def main():
    strict = json.load(open(HERE / "egpteval_results.json"))
    have = set()
    for m in ("judge_raw_egpteval2.jsonl", "judge_raw_egretry1_g2.jsonl"):
        p = HERE / m
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            have.add(json.loads(line)["key"].split("#")[0])
    # collect parsed strict rows per base key from existing aggregate is implicit;
    # rebuild: strict rows = all parsed (1934). Recover from failed_keys:
    rows = {}
    strict_rx = re.compile(r"^\s*Grammar:\s*(\d{1,2})\s*/\s*10\s*,\s*Creativity:\s*(\d{1,2})"
                           r"\s*/\s*10\s*,\s*Consistency:\s*(\d{1,2})\s*/\s*10\s*,"
                           r"\s*Age group:\s*([A-F])\b")
    def get_rows():
        for path in [HERE / "judge_raw_egpteval2.jsonl", HERE / "judge_raw_egretry1_g2.jsonl"]:
            if not path.exists():
                continue
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                yield r["key"].split("#")[0], r
    for base, r in get_rows():  # strict first (protocol)
        if base in rows or r.get("error"):
            continue
        m = strict_rx.search((r["response_text"] or "").replace("**", ""))
        if not m:
            continue
        g, c, s, a = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        rows[base] = {"g": g, "c": c, "s": s, "age": a, "lenient": False}
    for base, r in get_rows():  # lenient only for strict leftovers
        if base in rows or r.get("error"):
            continue
        m = LENIENT.search(r["response_text"] or "")
        if not m:
            continue
        g, c, s, a = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        if 0 <= g <= 10 and 0 <= c <= 10 and 0 <= s <= 10:
            rows[base] = {"g": g, "c": c, "s": s, "age": a, "lenient": True}
    n_lenient = sum(1 for v in rows.values() if v["lenient"])
    still_missing = [k for k in strict["failed_keys"] if k not in rows]
    print(f"total scored={len(rows)} (lenient-recovered={n_lenient}) still-missing={len(still_missing)}")
    print("still missing:", still_missing)
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
        from collections import Counter
        res["age_dist"] = {a: ages.count(a) for a in "ABCDEF"}
        res["n_prompts"] = len(pids)
        res["n_scored"] = sum(len(per[(m, p)]) for p in pids)
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
    json.dump({"models": out, "paired": pairs, "n_scored": len(rows),
               "n_lenient": n_lenient, "still_missing": still_missing},
              open(HERE / "egpteval_results_lenient.json", "w"), indent=2)
    print("wrote egpteval_results_lenient.json")
    # zero-imputed variant: N/A refusals (degenerate, scoreless) count as nominal 0s
    for base in still_missing:
        _, model, pid, ci = base.split("|")
        rows[base] = {"g": 0, "c": 0, "s": 0, "age": None, "lenient": False, "zero_na": True}
    per2 = {}
    for base, v in rows.items():
        _, model, pid, ci = base.split("|")
        per2.setdefault((model, int(pid)), []).append(v)
    rng2 = np.random.default_rng(7)
    out2 = {}
    for m in models:
        pids = sorted(set(p for (mm, p) in per2 if mm == m))
        means = {k: [] for k in ("g", "c", "s")}
        ages = []
        for p in pids:
            vs = per2[(m, p)]
            for k in means:
                means[k].append(float(np.mean([v[k] for v in vs])))
            ages.extend(v["age"] for v in vs if v["age"] is not None)
        res = {}
        for k, vals in means.items():
            v = np.asarray(vals)
            bs = np.mean(rng2.choice(v, size=(10000, len(v)), replace=True), axis=1)
            res[k] = {"mean": float(v.mean()), "lo": float(np.percentile(bs, 2.5)),
                      "hi": float(np.percentile(bs, 97.5))}
        from collections import Counter as Counter2
        res["age_dist"] = {a: ages.count(a) for a in "ABCDEF"}
        res["n_prompts"] = len(pids)
        res["n_scored"] = sum(len(per2[(m, p)]) for p in pids)
        out2[m] = res
    pairs2 = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            common = sorted(set(p for (mm, p) in per2 if mm == a) &
                            set(p for (mm, p) in per2 if mm == b))
            d = {}
            for k in ("g", "c", "s"):
                va = np.array([np.mean([v[k] for v in per2[(a, p)]]) for p in common])
                vb = np.array([np.mean([v[k] for v in per2[(b, p)]]) for p in common])
                diff = va - vb
                bs = np.mean(rng2.choice(diff, size=(10000, len(diff)), replace=True), axis=1)
                d[k] = {"mean": float(diff.mean()), "lo": float(np.percentile(bs, 2.5)),
                        "hi": float(np.percentile(bs, 97.5))}
            pairs2[f"{a}_vs_{b}"] = d
    json.dump({"models": out2, "paired": pairs2, "n_scored": len(rows),
               "note": "5 N/A refusals imputed as nominal 0s (worst case)"},
              open(HERE / "egpteval_results_full2000.json", "w"), indent=2)
    print("wrote egpteval_results_full2000.json (2000/2000 with zero-imputed N/A)")


if __name__ == "__main__":
    main()
