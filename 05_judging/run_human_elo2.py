#!/usr/bin/env python3
import csv
import json
import math
import os
import sys
from collections import defaultdict

MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]

def load_csv(path):
    res = []
    if not os.path.exists(path):
        print(f"WARN: {path} not found", file=sys.stderr)
        return res
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            res.append(r)
    print(f"Loaded {len(res)} rows from {path}", file=sys.stderr)
    return res

def main():
    try:
        import numpy as np
        from scipy.optimize import minimize
    except Exception as e:
        print(f"ERROR: need scipy/numpy: {e}", file=sys.stderr)
        sys.exit(1)

    h33 = load_csv("../cloud_training/eval/eval_outputs/human33m_ratings.csv")
    h50 = load_csv("../cloud_training/eval/eval_outputs/human50_ratings.csv")
    jv  = load_csv("second_judge_out/verdicts.csv")

    raw = []
    for r in h33 + h50:
        A = r.get("shown_first") or ""
        B = r.get("shown_second") or ""
        w = (r.get("winner") or "").strip()
        if A not in MODELS or B not in MODELS: continue
        s = 0.5
        if w == A: s = 1.0
        elif w == B: s = 0.0
        else: s = 0.5
        raw.append((MODELS.index(A), MODELS.index(B), s, A, B))

    print(f"\nOrder-level human items: {len(raw)}")
    cnt = defaultdict(int)
    for _,_,s,A,B in raw:
        cnt[(A,B,"A" if s==1.0 else ("B" if s==0.0 else "TIE"))]+=1
    for k,v in sorted(cnt.items())[:10]:
        print(f"  {k[0]} vs {k[1]}: {k[2]}={v}")
    if len(cnt)>10: print(f"  ... {len(cnt)-10} more pairs")

    def nll_raw(q):
        th = np.array([q[0], q[1], 0.0, q[2]])
        g  = q[3]
        tot = 0.0
        for X,Y,s,_,_ in raw:
            z = th[X]-th[Y]+g
            p = 1.0/(1.0+np.exp(-z))
            tot -= s*math.log(p+1e-18) + (1-s)*math.log(1-p+1e-18)
        return tot

    resr = minimize(nll_raw, np.zeros(4), method="BFGS", options={"gtol":1e-10})
    th2 = np.array([resr.x[0], resr.x[1], 0.0, resr.x[2]])
    g2  = float(resr.x[3])
    elo2 = {m: 1000.0+400.0/math.log(10.0)*(th2[i]-th2[2]) for i,m in enumerate(MODELS)}
    print("\n=== HUMAN BT (raw + order term), ts_1M anchored ===")
    print(json.dumps({"elo": {k: round(v,1) for k,v in elo2.items()}, "gamma": round(g2,3), "P_slotA_equal": round(1.0/(1.0+math.exp(-g2)),3), "converged": resr.success, "n": len(raw)}, indent=2))

    if jv:
        jp = defaultdict(lambda: {"A":0,"B":0,"TIE":0})
        for r in jv:
            try:
                a,b = r["pair"].split(" vs ")
                jp[(a,b)][r["parsed_verdict"]] += 1
            except Exception:
                pass
        print("\n=== SECOND JUDGE (for comparison) ===")
        for k,v in sorted(jp.items()):
            n=v["A"]+v["B"]+v["TIE"]
            print(f"  {k[0]} vs {k[1]}: A={v['A']} B={v['B']} TIE={v['TIE']} (n={n})")

if __name__=="__main__":
    main()
