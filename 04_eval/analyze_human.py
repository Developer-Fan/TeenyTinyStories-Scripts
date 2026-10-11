"""Analyze completed human ratings. Usage:
  python analyze_human.py human_ratings_<rater>.csv [more...]
Reads eval_outputs/{selected_50,human_packet}.csv + judge elo_results.json.
Writes eval_outputs/human_analysis.json (+ prints summary).
"""
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"
MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
ANCHOR = "ts_1M"


def load_packet():
    rows = list(csv.DictReader(open(OUT / "human_packet.csv")))
    return {(int(r["prompt_id"]),): r for r in rows}


def load_judge():
    elo = json.load(open(OUT / "elo_results.json"))
    return elo


def kappa(a, b, labels=("A", "B", "TIE")):
    a = list(a)
    b = list(b)
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == L) / n) * (sum(1 for y in b if y == L) / n)
             for L in labels)
    return (po - pe) / max(1 - pe, 1e-9), po


def bt_elos(pairs):
    import numpy as _np
    mi = {m: i for i, m in enumerate(MODELS)}
    ia = _np.array([mi[a] for a, b, wa, wb in pairs])
    ib = _np.array([mi[b] for a, b, wa, wb in pairs])
    wa = _np.array([wa for a, b, wa, wb in pairs])
    wt = _np.array([wa + wb for a, b, wa, wb in pairs])
    th = _np.zeros(len(MODELS))
    n = max(len(pairs), 1)
    for _ in range(300):
        pa = 1 / (1 + _np.exp(-(th[ia] - th[ib])))
        d = wa - wt * pa
        g = _np.bincount(ia, weights=d, minlength=len(MODELS)) - \
            _np.bincount(ib, weights=d, minlength=len(MODELS))
        th += 0.5 * (g / n - 0.005 * th / len(MODELS))
        th -= th.mean()
    return {m: 1000 + (400 / math.log(10)) * float(th[mi[m]] - th[mi[ANCHOR]])
            for m in MODELS}


def main():
    packet = load_packet()
    sel = list(csv.DictReader(open(OUT / "selected_50.csv")))
    selmap = {(int(r["prompt_id"]),): (r["model_a"], r["model_b"]) for r in sel}
    # judge consolidated verdicts for the same 50 (from pairwise raw)
    import re
    RX = re.compile(r"^\s*Verdict:\s*(A|B|TIE)\s*$", re.IGNORECASE | re.MULTILINE)
    jv = {}
    with open(OUT / "judge_raw_pairwise.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                continue
            ms = RX.findall(r["response_text"] or "")
            if ms and r["key"] not in jv:
                jv[r["key"]] = ms[-1].upper()
    # raters
    raters = {}
    for path in sys.argv[1:]:
        for r in csv.DictReader(open(path)):
            raters.setdefault(r["rater_id"], {})[int(r["item"])] = r["verdict"].upper()
    print(f"raters={list(raters)} items=50")
    packet_list = sorted(packet.values(), key=lambda r: int(r["prompt_id"]))
    # per-rater verdicts mapped to model wins; majority vote
    votes_item = {}
    for rater, ans in raters.items():
        for item, v in ans.items():
            p = packet_list[item]
            a_wins = {"A": p["first_is"], "B": p["second_is"], "TIE": "TIE"}[v]
            votes_item.setdefault(item, []).append(a_wins)
    majority = {}
    for item, vs in votes_item.items():
        majority[item] = max(set(vs), key=vs.count)
    # judge verdicts on same items (order-consolidated)
    judge50 = {}
    for item in range(50):
        p = packet_list[item]
        pid = int(p["prompt_id"])
        a, b = p["first_is"], p["second_is"]
        # find original pair key orientation
        ma, mb = selmap[(pid,)]
        ka = f"p|{ma}|{mb}|{pid}|ab"
        kb = f"p|{ma}|{mb}|{pid}|ba"
        va, vb = jv.get(ka), jv.get(kb)
        if va is None or vb is None:
            judge50[item] = None
            continue
        wa = "A" if (va == "A") == (ma == a) else ("B" if va != "TIE" else "TIE")
        # simpler: map each order verdict to model, then consolidate like aggregate
        def towin(v, first_is):
            if v == "TIE":
                return "TIE"
            return first_is if v == "A" else (mb if first_is == ma else ma)
        w1, w2 = towin(va, ma), towin(vb, mb)
        judge50[item] = w1 if w1 == w2 and w1 != "TIE" else "TIE"
    # agreement human-majority vs judge (drop Nones)
    items = [i for i in range(50) if judge50.get(i) is not None and i in majority]
    hv = [majority[i] for i in items]
    jv50 = [judge50[i] for i in items]
    k_all, po_all = kappa(hv, jv50)
    notie = [(h, j) for h, j in zip(hv, jv50) if h != "TIE" and j != "TIE"]
    k_nt, po_nt = kappa([h for h, _ in notie], [j for _, j in notie]) if notie else (0, 0)
    print(f"human-vs-judge: n={len(items)} agree={po_all:.2f} kappa={k_all:.2f} | "
          f"no-tie subset n={len(notie)} agree={po_nt:.2f} kappa={k_nt:.2f}")
    # inter-rater
    rids = sorted(raters)
    if len(rids) >= 2:
        for x, y in itertools.combinations(rids, 2):
            common = sorted(set(raters[x]) & set(raters[y]))
            k, po = kappa([raters[x][i] for i in common], [raters[y][i] for i in common])
            print(f"inter-rater {x}-{y}: agree={po:.2f} kappa={k:.2f}")
    # human Elo on 50 (majority verdicts)
    hpairs = []
    for item in range(50):
        if item not in majority:
            continue
        p = packet_list[item]
        w = majority[item]
        a, b = p["first_is"], p["second_is"]
        hpairs.append((a, b, 1.0 if w == a else (0.0 if w == b else 0.5),
                       1.0 if w == b else (0.0 if w == a else 0.5)))
    helo = bt_elos([(a, b, wa, wb) for a, b, wa, wb in hpairs])
    rng = np.random.default_rng(5)
    boots = {m: [] for m in MODELS}
    for _ in range(2000):
        idx = rng.choice(len(hpairs), size=len(hpairs), replace=True)
        eb = bt_elos([hpairs[i] for i in idx])
        for m in MODELS:
            boots[m].append(eb[m])
    hci = {m: {"elo": helo[m], "lo": float(np.percentile(boots[m], 2.5)),
              "hi": float(np.percentile(boots[m], 97.5))} for m in MODELS}
    # judge on all-300 vs same-50
    full = json.load(open(OUT / "elo_results.json"))["elo"]
    # judge-50: refit BT on the 50 consolidated verdicts
    jpairs = []
    for item in range(50):
        if judge50.get(item) is None:
            continue
        p = packet_list[item]
        w = judge50[item]
        a, b = p["first_is"], p["second_is"]
        jpairs.append((a, b, 1.0 if w == a else (0.0 if w == b else 0.5),
                       1.0 if w == b else (0.0 if w == a else 0.5)))
    j50 = bt_elos(jpairs)
    def ranks(d):
        s = sorted(d, key=d.get)
        return {m: i for i, m in enumerate(s)}

    def spearman(r1, r2):
        n = len(r1)
        d2 = sum((r1[m] - r2[m]) ** 2 for m in r1)
        return 1 - 6 * d2 / (n * (n * n - 1))
    rh, rj, rf = ranks(helo), ranks(j50), ranks({m: full[m]["elo"] for m in MODELS})
    print("human-50 elo:", {m: round(v) for m, v in helo.items()})
    print("judge-50 elo:", {m: round(v) for m, v in j50.items()})
    print("judge-300 elo:", {m: round(full[m]['elo']) for m in MODELS})
    print("rank orders:", rh, rj, rf)
    print("spearman h-j50:", round(spearman(rh, rj), 3),
          "h-j300:", round(spearman(rh, rf), 3),
          "j50-j300:", round(spearman(rj, rf), 3))
    json.dump({"majority": {str(k): v for k, v in majority.items()},
               "human_elo": hci, "judge50_elo": j50,
               "agreement": {"n": len(items), "po": po_all, "kappa": k_all,
                             "no_tie_n": len(notie), "no_tie_po": po_nt,
                             "no_tie_kappa": k_nt},
               "rank_orders": {"human50": rh, "judge50": rj, "judge300": rf}},
              open(OUT / "human_analysis.json", "w"), indent=2)
    print("wrote human_analysis.json (50 items: wide CIs expected)")


if __name__ == "__main__":
    main()
