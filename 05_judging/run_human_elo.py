#!/usr/bin/env python3
"""Run Bradley-Terry Elo analysis on human ratings, matching the pairwise judge's Elo computation where reasonable.

This script expects human ratings files to exist in the analyse directory:
- human33m_ratings.csv
- human50_ratings.csv

It computes Elo for the 4 models (ours_1M, ours_33M, ts_1M, ts_33M) by extracting pairwise/score comparisons
from the human ratings (where available) and fits a Bradley-Terry model (unregularized MLE) via scipy.optimize,
anchoring ts_1M at 1000. If the data do not contain pairwise outcomes for all 6 pairs, it still reports
what is available and leaves others undefined.

It also optionally can compare against judge verdicts if verdicts.csv is present.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]

def load_human33m(path="analyse/human33m_ratings.csv"):
    res = []
    if not os.path.exists(path):
        print(f"WARN: {path} not found", file=sys.stderr)
        return res
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            res.append(r)
    print(f"Loaded {len(res)} rows from {path}", file=sys.stderr)
    return res

def load_human50(path="analyse/human50_ratings.csv"):
    res = []
    if not os.path.exists(path):
        print(f"WARN: {path} not found", file=sys.stderr)
        return res
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            res.append(r)
    print(f"Loaded {len(res)} rows from {path}", file=sys.stderr)
    return res

def load_judge_verdicts(path="analyse/second_judge_out/verdicts.csv"):
    res = []
    if not os.path.exists(path):
        print(f"WARN: {path} not found", file=sys.stderr)
        return res
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            res.append(r)
    print(f"Loaded {len(res)} rows from {path}", file=sys.stderr)
    return res

def try_scipy():
    try:
        import numpy as np
        from scipy.optimize import minimize
        return np, minimize
    except Exception as e:
        print(f"ERROR: scipy/numpy not available: {e}", file=sys.stderr)
        return None, None

def sigmoid(x):
    # avoid overflow
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)

def bt_elo(items, seed=0, nboot=2000):
    """items: list of (i,j,s) with i!=j, s in [0,1] (1.0 if i wins, 0.5 tie, 0.0 if j wins).
    Return dict with elo and optional bootstrap CIs if scipy available."""
    np, minimize = try_scipy()
    if np is None:
        return {"error": "scipy/numpy required for BT fit"}

    MODELI = MODELS
    idx = {m:k for k,m in enumerate(MODELI)}

    # build consolidated items per (i,j) if we had prompt structure? but human files unknown; use items as is
    def nll(q):
        th = np.array([q[0], q[1], 0.0, q[2]])  # anchor index 2 (ts_1M)
        tot = 0.0
        for i,j,s in items:
            z = th[i]-th[j]
            p = 1.0/(1.0+np.exp(-z))
            if s == 0.5:
                tot -= math.log(p*(1-p) + 1e-18)  # tie prob approx via Bradley-Terry with draw? but standard: s*log p + (1-s)*log(1-p) with p=sigmoid(z)
            # use standard
        return tot

    # standard nll
    def nll2(q):
        th = np.array([q[0], q[1], 0.0, q[2]])
        tot = 0.0
        for i,j,s in items:
            z = th[i]-th[j]
            p = 1.0/(1.0+np.exp(-z))
            tot -= s*math.log(p+1e-18) + (1-s)*math.log(1-p+1e-18)
        return tot

    res = minimize(nll2, np.zeros(3), method="BFGS", options={"gtol":1e-10})
    th = np.array([res.x[0], res.x[1], 0.0, res.x[2]])
    elo = {m: 1000.0 + 400.0/math.log(10.0)*(th[k]-th[2]) for k,m in enumerate(MODELI)}
    out = {"elo": {m: round(float(e),1) for m,e in elo.items()}, "converged": bool(res.success), "n_items": len(items)}

    # simple bootstrap if items look grouped? skip heavy bootstrap unless grouped; just note
    return out

def main():
    print("=== HUMAN ELO EVAL RUNNER ===")
    h33 = load_human33m("human33m_ratings.csv")
    h50 = load_human50("human50_ratings.csv")
    jv = load_judge_verdicts("second_judge_out/verdicts.csv")

    # Try to infer pairwise outcomes from human files.
    # Common patterns: columns like model_a, model_b, winner, score_a, score_b, prompt_id
    items = []  # (i,j,s)
    def add_item(a,b,winner=None, score_a=None, score_b=None):
        if a not in MODELS or b not in MODELS:
            return
        i = MODELS.index(a); j = MODELS.index(b)
        if i>j:
            a,b,i,j = b,a,j,i
            if winner == "a": winner="b"
            elif winner == "b": winner="a"
            if score_a is not None and score_b is not None:
                score_a,score_b = score_b,score_a
        s = 0.5
        if winner == "a": s=1.0
        elif winner == "b": s=0.0
        elif score_a is not None and score_b is not None:
            sa=float(score_a); sb=float(score_b)
            if sa > sb+1e-9: s=1.0
            elif sb > sa+1e-9: s=0.0
            else: s=0.5
        items.append((i,j,s))

    for r in h33+h50:
        a=r.get("model_a") or r.get("a") or r.get("model1")
        b=r.get("model_b") or r.get("b") or r.get("model2")
        winner=r.get("winner") or r.get("pref")
        sa=r.get("score_a") or r.get("score1") or r.get("rating_a")
        sb=r.get("score_b") or r.get("score2") or r.get("rating_b")
        if a and b:
            add_item(a,b,winner, sa,sb)

    print(f"\nInferred pairwise items from human ratings: {len(items)}")
    if items:
        cnt = defaultdict(int)
        for i,j,s in items:
            cnt[(MODELS[i],MODELS[j], "A" if s==1.0 else ("B" if s==0.0 else "TIE"))]+=1
        for k,v in sorted(cnt.items()):
            print(f"  {k[0]} vs {k[1]}: {k[2]}={v}")
    else:
        print("  (no pairwise outcomes inferred; check column names)")

    np,minimize = try_scipy()
    if np is None:
        print("\nCannot fit BT without scipy/numpy. Install scipy to get Elo.")
        sys.exit(0)

    res = bt_elo(items)
    print("\n=== HUMAN BT ELO (unregularized, ts_1M anchored) ===")
    print(json.dumps(res, indent=2))

    # Compare to judge if desired: quick head-to-head summary
    if jv:
        print("\n=== JUDGE VERDICTS (second judge) SUMMARY ===")
        # count per pair
        jp = defaultdict(lambda: {"A":0,"B":0,"TIE":0})
        for r in jv:
            a,b = r["pair"].split(" vs ")
            jp[(a,b)][r["parsed_verdict"]]+=1
        for k,v in sorted(jp.items()):
            print(f"  {k[0]} vs {k[1]}: A={v['A']} B={v['B']} TIE={v['TIE']} (n={v['A']+v['B']+v['TIE']})")

if __name__=="__main__":
    main()
