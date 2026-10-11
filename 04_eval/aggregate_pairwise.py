"""Aggregate pairwise: parse verdicts (LAST matching Verdict line), consolidate
orders (agree->winner else tie), Bradley-Terry -> Elo anchored ts_1M=1000,
bootstrap CIs over prompts, win-rate matrix + transitivity, length check.
Usage: python aggregate_pairwise.py
"""
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"
RX = re.compile(r"^\s*Verdict:\s*(A|B|TIE)\s*$", re.IGNORECASE | re.MULTILINE)
MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
ANCHOR = "ts_1M"


def load_raw():
    out = {}
    with open(OUT / "judge_raw_pairwise.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                continue
            ms = RX.findall((r["response_text"] or "").replace("**", ""))
            if not ms:
                continue
            v = ms[-1].upper()
            if r["key"] not in out:  # first success wins
                out[r["key"]] = v
    return out


def consolidate(votes):
    """votes: {(a,b,pid): {'ab': v|'ab' missing, 'ba': ...}} -> (w_a, w_b)."""
    res = {}
    for k, d in votes.items():
        va, vb = d.get("ab"), d.get("ba")
        if va is None or vb is None:
            continue  # incomplete pair: both orders required
        a, b, pid = k
        # map order-verdicts to model wins: in 'ab', A=a; in 'ba', A=b
        wa = wb = 0.0
        for order, v in (("ab", va), ("ba", vb)):
            first_is_a = (order == "ab")
            if v == "TIE":
                wa += 0.25
                wb += 0.25
            elif (v == "A") == first_is_a:
                wa += 0.5
            else:
                wb += 0.5
        # agree-on-winner check for stats handled separately; score = mean of orders
        res[k] = (wa, wb)
    return res


def bt_fit(pairs, models, l2=0.005, iters=4000, lr=0.5):  # weak prior: near-sweeps would otherwise blow up unregularized; strong priors shrink gaps (see README)
    """Bradley-Terry via gradient ascent (numpy-vectorized). Ties = half wins."""
    import numpy as _np
    mi = {m: i for i, m in enumerate(models)}
    ia = _np.array([mi[a] for a, b, wa, wb in pairs])
    ib = _np.array([mi[b] for a, b, wa, wb in pairs])
    wa = _np.array([wa for a, b, wa, wb in pairs])
    wt = _np.array([wa + wb for a, b, wa, wb in pairs])
    th = _np.zeros(len(models))
    n = max(len(pairs), 1)
    for _ in range(iters):
        pa = 1 / (1 + _np.exp(-(th[ia] - th[ib])))
        d = wa - wt * pa
        g = _np.bincount(ia, weights=d, minlength=len(models)) - \
            _np.bincount(ib, weights=d, minlength=len(models))
        th += lr * (g / n - l2 * th / len(models))
        th -= th.mean()
    return {m: float(th[mi[m]]) for m in models}


def to_elo(th):
    return {m: 1000 + (400 / math.log(10)) * (th[m] - th[ANCHOR]) for m in th}


def main():
    votes_raw = load_raw()
    print(f"parsed verdicts: {len(votes_raw)}/600")
    votes = {}
    firstwin = {"A": 0, "B": 0, "TIE": 0}
    agree = tie = inconsist = 0
    for key, v in votes_raw.items():
        _, a, b, pid, order = key.split("|")
        votes.setdefault((a, b, int(pid)), {})[order] = v
        firstwin[v] += 1
    for (a, b, pid), d in votes.items():
        if "ab" not in d or "ba" not in d:
            continue
        # winner in each order mapped to model
        w_ab = a if d["ab"] == "A" else (b if d["ab"] == "B" else "TIE")
        w_ba = b if d["ba"] == "A" else (a if d["ba"] == "B" else "TIE")
        if w_ab == w_ba and w_ab != "TIE":
            agree += 1
        elif w_ab == "TIE" or w_ba == "TIE":
            tie += 1
        else:
            inconsist += 1
    n = agree + tie + inconsist
    print(f"order-complete={n} agree={agree} tie-or-partial={tie} disagree={inconsist}")
    print(f"first-listed wins: A={firstwin['A']} B={firstwin['B']} TIE={firstwin['TIE']}")
    res = consolidate(votes)
    pairs = [(a, b, wa, wb) for (a, b, _), (wa, wb) in res.items()]
    th = bt_fit(pairs, MODELS)
    elo = to_elo(th)
    # bootstrap over prompts (resample with replacement, refit, re-anchor)
    rng = np.random.default_rng(21)
    pids = sorted(set(pid for _, _, pid in res))
    by_pid = {}
    for (a, b, p), (wa, wb) in res.items():
        by_pid.setdefault(p, []).append((a, b, wa, wb))
    boots = {m: [] for m in MODELS}
    for _ in range(2000):
        samp = rng.choice(pids, size=len(pids), replace=True)
        sp = []
        for p in samp:
            sp.extend(by_pid[p])
        eb = to_elo(bt_fit(sp, MODELS))
        for m in MODELS:
            boots[m].append(eb[m])
    ci = {m: {"elo": elo[m], "lo": float(np.percentile(boots[m], 2.5)),
              "hi": float(np.percentile(boots[m], 97.5))} for m in MODELS}
    # win matrix
    wm = {}
    for a, b in itertools.combinations(MODELS, 2):
        ws = [res[k] for k in res if k[0] == a and k[1] == b]
        wa = sum(x[0] for x in ws)
        wb = sum(x[1] for x in ws)
        wm[f"{a}_vs_{b}"] = {"a_wins": wa, "b_wins": wb, "n": len(ws),
                             "a_rate": wa / max(len(ws), 1)}
    # transitivity: majority winner per pair, then look for cycles
    def W(x, y):
        w = wm[f"{x}_vs_{y}"] if f"{x}_vs_{y}" in wm else None
        if w is None:
            w = wm[f"{y}_vs_{x}"]
            return y if w["a_wins"] > w["b_wins"] else x
        return x if w["a_wins"] > w["b_wins"] else y

    cyc = []
    for a, b, c in itertools.combinations(MODELS, 3):
        if W(a, b) == b and W(b, c) == c and W(a, c) == a:
            cyc.append((a, b, c))
    # length check
    gens = {}
    with open(OUT / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["completion_idx"] == 0:
                gens[(r["model"], r["prompt_id"])] = r["n_tokens"]
    ld, wo = [], []
    for (a, b, pid), (wa, wb) in res.items():
        la, lb = gens[(a, pid)], gens[(b, pid)]
        ld.append(la - lb)
        wo.append(1 if wa > wb else (-1 if wb > wa else 0))
    ld = np.asarray(ld, float)
    wo = np.asarray(wo, float)
    corr = float(np.corrcoef(ld, wo)[0, 1]) if len(ld) > 2 else 0.0
    json.dump({"elo": ci, "win_matrix": wm, "cycles": cyc,
               "position": {"first_A": firstwin["A"], "first_B": firstwin["B"],
                            "ties": firstwin["TIE"],
                            "first_win_rate": (firstwin["A"] + firstwin["B"]) / max(sum(firstwin.values()), 1)},
               "order_stats": {"agree": agree, "tie_or_partial": tie,
                               "disagree": inconsist, "n": n},
               "length_corr": corr,
               "mean_len": {m: float(np.mean([gens[(m, p)] for p in range(1, 51)])) for m in MODELS}},
              open(OUT / "elo_results.json", "w"), indent=2)
    print("elo:", {m: round(v["elo"]) for m, v in ci.items()})
    print("wrote elo_results.json")


if __name__ == "__main__":
    main()
