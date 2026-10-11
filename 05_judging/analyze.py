#!/usr/bin/env python3
"""Step 4 export + Step 5 analysis for the second-judge run.

Reads (read-only): jobs_600.jsonl, calls_log.jsonl (this dir); generations.csv (analyse/);
  cloud_training/eval/eval_outputs/judge_raw_pairwise.jsonl (Ministral verdicts, item 10 only).
Writes: verdicts.csv, failures_log.txt, analysis.json (stdout = analysis text).

Definitions (fixed before results were seen):
  verdict  : strict parse of each order-level response (regex from aggregate_pairwise.py).
             Attempt 1 is used if it parses; otherwise attempt 2 (logged retry) if it parses.
  consolidated item (pair, prompt), both orders valid:
             both orders name the same model          -> 'consistent'   (winner 1, loser 0)
             orders name different models             -> 'order_inconsistent' (0.5 each)
             at least one order is TIE                -> 'TIE'          (0.5 each)
  Bradley-Terry: logistic on natural-log strengths, ts_1M fixed at 0; Elo = 1000 + (400/ln10)*theta.
             Main fit: unregularized MLE (Newton). If it separates (diverges), a ridge penalty
             of RIDGE on theta is used for that fit and the fit is labelled as such.
  Prompt bootstrap: 2000 resamples of the 50 prompt ids (with replacement), seed 0.
  Sensitivity: BT on all valid order-level verdicts (TIE=0.5), with an order (slot-A) parameter.
"""
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSE = Path("/Users/yanfu/Documents/TeenyTinyStories/analyse")
MIN_RAW = Path("/Users/yanfu/Documents/TeenyTinyStories/cloud_training/eval/eval_outputs/judge_raw_pairwise.jsonl")
MODEL_REQ = "infron:openai/gpt-5.6-luna"

MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
PAIRS = list(itertools.combinations(MODELS, 2))
PIDS = list(range(1, 51))
ANCHOR = "ts_1M"
AI = MODELS.index(ANCHOR)
ELO_SCALE = 400 / math.log(10)
RIDGE = 0.01
BOOT_N = 2000
BOOT_SEED = 0
Z = 1.959963984540054
RX = re.compile(r"^\s*Verdict:\s*(A|B|TIE)\s*$", re.IGNORECASE | re.MULTILINE)


def parse(text):
    if text is None:
        return None, "NO_TEXT"
    ms = RX.findall(text.replace("**", ""))
    if not ms:
        return None, "NO_MATCH"
    return ms[-1].upper(), "OK"


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- load and resolve verdicts
jobs = []
with (HERE / "jobs_600.jsonl").open(encoding="utf-8") as f:
    for l in f:
        jobs.append(json.loads(l))
jobs_by_key = {j["key"]: j for j in jobs}

calls = defaultdict(list)
with (HERE / "calls_log.jsonl").open(encoding="utf-8") as f:
    for l in f:
        if l.strip():
            r = json.loads(l)
            calls[r["key"]].append(r)

resolved = {}          # key -> dict(verdict, rec, status_by_attempt)
failures = []          # every failed attempt
retry_used = []        # keys where attempt 2 was made
retry_when_att1_ok = []
for j in jobs:
    key = j["key"]
    atts = sorted(calls.get(key, []), key=lambda r: r["attempt"])
    chosen = None
    status = []
    for r in atts:
        if r["attempt"] == 2:
            retry_used.append(key)
        if not r["ok"]:
            status.append((r["attempt"], "API_ERROR"))
            failures.append((key, r, "API_ERROR"))
            continue
        v, st = parse(r["response_text"])
        status.append((r["attempt"], st))
        if st != "OK":
            failures.append((key, r, st))
        elif chosen is None:
            chosen = (v, r)
    if any(r["attempt"] == 2 for r in atts) and any(s[0] == 1 and s[1] == "OK" for s in status):
        retry_when_att1_ok.append(key)
    last = atts[-1] if atts else None
    if chosen:
        resolved[key] = {"verdict": chosen[0], "rec": chosen[1], "status": status, "last": last}
    else:
        resolved[key] = {"verdict": None, "rec": None, "status": status, "last": last}

n_att = {k: len(calls.get(k, [])) for k in jobs_by_key}
n_valid = sum(1 for v in resolved.values() if v["verdict"] is not None)
verdict_of = {k: v["verdict"] for k, v in resolved.items()}


def winner_of(v, a, b, order):
    if v is None:
        return None
    if v == "TIE":
        return "TIE"
    slotA, slotB = (a, b) if order == "ab" else (b, a)
    return slotA if v == "A" else slotB


# ---------------------------------------------------------------- verdicts.csv and failures_log.txt
with (HERE / "verdicts.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["prompt_id", "pair", "order", "key", "A_model", "B_model", "attempts_logged",
                "attempt_used", "parse_status_by_attempt", "parsed_verdict", "winner_model",
                "model_requested", "model_returned", "utc_timestamp", "latency_ms", "raw_output"])
    for j in jobs:
        t = j["tag"]
        key = j["key"]
        rv = resolved[key]
        used = rv["rec"]
        shown = used if used is not None else rv["last"]
        slotA, slotB = (t["a"], t["b"]) if t["order"] == "ab" else (t["b"], t["a"])
        v = rv["verdict"]
        w.writerow([t["pid"], f"{t['a']} vs {t['b']}", t["order"], key, slotA, slotB,
                    n_att[key], used["attempt"] if used else "NONE",
                    ";".join(f"a{a}:{s}" for a, s in rv["status"]),
                    v if v is not None else "FAILED",
                    winner_of(v, t["a"], t["b"], t["order"]) or "FAILED",
                    MODEL_REQ,
                    (shown or {}).get("model_returned") if shown and shown.get("model_returned") is not None else "NOT_REPORTED_BY_API",
                    (shown or {}).get("utc_timestamp", ""),
                    (shown or {}).get("latency_ms", ""),
                    (shown or {}).get("response_text") if shown and shown.get("response_text") is not None
                    else (shown or {}).get("error", "")])

with (HERE / "failures_log.txt").open("w", encoding="utf-8") as f:
    f.write(f"failed attempts logged: {len(failures)}\n")
    for key, r, kind in failures:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"{key} attempt={r['attempt']} phase={r['phase']} kind={kind} utc={r['utc_timestamp']}\n")
        if kind == "API_ERROR":
            f.write(f"error: {r['error']}\n")
        else:
            f.write(f"finish_reason: {r.get('finish_reason')}\nraw output:\n{r.get('response_text')!r}\n")

# ---------------------------------------------------------------- consolidation
verdict_by = {}  # (a,b,pid,order) -> verdict
for j in jobs:
    t = j["tag"]
    verdict_by[(t["a"], t["b"], t["pid"], t["order"])] = verdict_of[j["key"]]

S = np.zeros((len(PAIRS), 50))  # s_a per item (0 where invalid)
V = np.zeros((len(PAIRS), 50))  # 1 if item complete (both orders valid)
cat = {}                        # (pi, pid) -> category
for pi, (a, b) in enumerate(PAIRS):
    for pid in PIDS:
        vab = verdict_by.get((a, b, pid, "ab"))
        vba = verdict_by.get((a, b, pid, "ba"))
        if vab is None or vba is None:
            continue
        wab = winner_of(vab, a, b, "ab")
        wba = winner_of(vba, a, b, "ba")
        if wab == "TIE" or wba == "TIE":
            c = "TIE_both" if (wab == "TIE" and wba == "TIE") else "TIE_one"
            s = 0.5
            cat[(pi, pid)] = "TIE" if c.startswith("TIE") else c
            cat[(pi, pid, "sub")] = c
        elif wab == wba:
            c = "consistent"
            s = 1.0 if wab == a else 0.0
            cat[(pi, pid)] = c
        else:
            c = "order_inconsistent"
            s = 0.5
            cat[(pi, pid)] = c
        S[pi, pid - 1] = s
        V[pi, pid - 1] = 1.0
n_complete = int(V.sum())


# ---------------------------------------------------------------- Bradley-Terry fits
def fit_bt(W, n, ridge=0.0):
    th = np.zeros(4)
    free = [0, 1, 3]
    status = "diverged"
    for _ in range(300):
        g = np.zeros(4)
        H = np.zeros((4, 4))
        for k, (a, b) in enumerate(PAIRS):
            if n[k] == 0:
                continue
            i, j = MODELS.index(a), MODELS.index(b)
            p = sig(th[i] - th[j])
            d = W[k] - n[k] * p
            wt = n[k] * p * (1 - p)
            g[i] += d
            g[j] -= d
            H[i, i] -= wt
            H[j, j] -= wt
            H[i, j] += wt
            H[j, i] += wt
        g -= ridge * th
        H -= ridge * np.eye(4)
        gf = g[free]
        Hf = H[np.ix_(free, free)]
        try:
            step = np.linalg.solve(Hf, gf)
        except np.linalg.LinAlgError:
            return th, "singular"
        th[free] = th[free] - step
        if np.max(np.abs(th)) > 30:
            return th, "diverged"
        if np.max(np.abs(gf)) < 1e-10:
            status = "converged"
            break
    return th, status


def main_fit(W, n):
    th, st = fit_bt(W, n, 0.0)
    if st == "converged":
        return th, "unregularized MLE"
    th2, st2 = fit_bt(W, n, RIDGE)
    return th2, f"ridge {RIDGE} (unregularized {st})"


def elo_of(th):
    return {m: 1000 + ELO_SCALE * (th[i] - th[AI]) for i, m in enumerate(MODELS)}


W_main = (S * V).sum(axis=1)
n_main = V.sum(axis=1)
th_main, method_main = main_fit(W_main, n_main)
elo_main = elo_of(th_main)

# prompt bootstrap, shared by all CIs
rng = np.random.default_rng(BOOT_SEED)
idx = rng.integers(0, 50, size=(BOOT_N, 50))
C = np.zeros((BOOT_N, 50))
np.add.at(C, (np.repeat(np.arange(BOOT_N), 50), idx.ravel()), 1)

Wb = C @ (S * V).T            # (BOOT_N, 6)
nb = C @ V.T
boot_elo = {m: [] for m in MODELS}
boot_methods = {"unregularized MLE": 0, "ridge": 0}
for r in range(BOOT_N):
    th_r, meth = main_fit(Wb[r], nb[r])
    boot_methods["ridge" if meth.startswith("ridge") else "unregularized MLE"] += 1
    er = elo_of(th_r)
    for m in MODELS:
        boot_elo[m].append(er[m])
elo_ci = {m: (float(np.percentile(boot_elo[m], 2.5)), float(np.percentile(boot_elo[m], 97.5))) for m in MODELS}


# sensitivity: raw verdicts with a slot-A (order) parameter
def fit_raw(recs):
    free = [0, 1, 3]
    q = np.zeros(4)  # theta for free models (ts_1M fixed), gamma
    status = "diverged"
    for _ in range(300):
        g = np.zeros(4)
        H = np.zeros((4, 4))
        for x, y, s in recs:
            z = np.zeros(4)
            if x != AI:
                z[free.index(x)] += 1
            if y != AI:
                z[free.index(y)] -= 1
            z[3] = 1.0
            p = sig(z @ q)
            g += (s - p) * z
            H -= p * (1 - p) * np.outer(z, z)
        step = np.linalg.solve(H, g)
        q = q - step
        if np.max(np.abs(q)) > 30:
            return q, "diverged"
        if np.max(np.abs(g)) < 1e-10:
            status = "converged"
            break
    return q, status


raw_recs = []
for j in jobs:
    t = j["tag"]
    v = verdict_of[j["key"]]
    if v is None:
        continue
    slotA, slotB = (t["a"], t["b"]) if t["order"] == "ab" else (t["b"], t["a"])
    s = 1.0 if v == "A" else (0.0 if v == "B" else 0.5)
    raw_recs.append((MODELS.index(slotA), MODELS.index(slotB), s))
q_raw, st_raw = fit_raw(raw_recs)
th_raw = np.zeros(4)
for i, m in enumerate([0, 1, 3]):
    th_raw[m] = q_raw[i]
gamma = q_raw[3]
elo_raw = elo_of(th_raw)

# ---------------------------------------------------------------- Ministral comparison input (item 10)
min_v = {}
with MIN_RAW.open(encoding="utf-8") as f:
    for l in f:
        r = json.loads(l)
        if r.get("error"):
            continue
        v, st = parse(r.get("response_text") or "")
        if st != "OK":
            continue
        if r["key"] not in min_v:
            min_v[r["key"]] = v

S_m = np.zeros((len(PAIRS), 50))
V_m = np.zeros((len(PAIRS), 50))
cat_m = {}
for pi, (a, b) in enumerate(PAIRS):
    for pid in PIDS:
        vab = min_v.get(f"p|{a}|{b}|{pid}|ab")
        vba = min_v.get(f"p|{a}|{b}|{pid}|ba")
        if vab is None or vba is None:
            continue
        wab = winner_of(vab, a, b, "ab")
        wba = winner_of(vba, a, b, "ba")
        if wab == "TIE" or wba == "TIE":
            cat_m[(pi, pid)] = "TIE"
            s = 0.5
        elif wab == wba:
            cat_m[(pi, pid)] = "consistent"
            s = 1.0 if wab == a else 0.0
        else:
            cat_m[(pi, pid)] = "order_inconsistent"
            s = 0.5
        S_m[pi, pid - 1] = s
        V_m[pi, pid - 1] = 1.0
th_m, method_m = main_fit((S_m * V_m).sum(axis=1), V_m.sum(axis=1))
elo_m = elo_of(th_m)

# ---------------------------------------------------------------- output
out = {}
print("=" * 78)
print("SECOND-JUDGE RUN: analysis output")
print("=" * 78)
print(f"model requested: {MODEL_REQ}  (model returned by API: none reported in any response)")
print(f"jobs: {len(jobs)}  valid verdicts: {n_valid}/{len(jobs)}  failed jobs: {len(jobs) - n_valid}")
print(f"attempts logged: {sum(n_att.values())}  failed attempts: {len(failures)}  "
      f"retries made: {len(retry_used)}  retries where attempt 1 was already valid: {len(retry_when_att1_ok)}")
print(f"complete (pair,prompt) items: {n_complete}/300")
missing = [j for j in jobs if verdict_of[j["key"]] is None]
with (HERE / "missing_verdicts.txt").open("w", encoding="utf-8") as f:
    f.write("prompt_id\tpair\torder\tkey\tattempts_logged\n")
    for j in missing:
        t = j["tag"]
        f.write(f"{t['pid']}\t{t['a']} vs {t['b']}\t{t['order']}\t{j['key']}\t{n_att[j['key']]}\n")
print(f"missing verdicts (no valid parse after attempts): {len(missing)}  -> missing_verdicts.txt")
missing = [j["key"] for j in missing]
out["counts"] = {"jobs": len(jobs), "valid": n_valid, "failed_jobs": missing,
                 "failed_attempts": len(failures), "retries": len(retry_used),
                 "complete_items": n_complete}

# item 1: position bias
valid_v = [v for v in verdict_of.values() if v is not None]
nB = sum(1 for v in valid_v if v == "B")
nA = sum(1 for v in valid_v if v == "A")
nT = sum(1 for v in valid_v if v == "TIE")
n_all = len(valid_v)
lo, hi = wilson(nB, n_all)
lod, hid = wilson(nB, nA + nB)
print("\n[1] Position bias (second-listed = slot B)")
print(f"  second-listed won: {nB}/{n_all} = {nB / n_all:.3f}  95% Wilson CI [{lo:.3f}, {hi:.3f}]")
print(f"  decisive only (A or B): {nB}/{nA + nB} = {nB / (nA + nB):.3f}  95% Wilson CI [{lod:.3f}, {hid:.3f}]")
print(f"  first-listed (A) won: {nA}/{n_all} = {nA / n_all:.3f}")
out["position"] = {"B": nB, "A": nA, "TIE": nT, "n": n_all, "B_share": nB / n_all, "B_ci": [lo, hi]}

# item 2: TIE
print("\n[2] Direct TIE verdicts")
print(f"  TIE: {nT}/{n_all} = {nT / n_all:.3f}")
out["ties"] = {"TIE": nT, "n": n_all}

# item 3: order-inconsistent pairs
n_oi = sum(1 for k, c in cat.items() if len(k) == 2 and c == "order_inconsistent")
n_tie_item = sum(1 for k, c in cat.items() if len(k) == 2 and c == "TIE")
print("\n[3] Order-inconsistent pairs (orders name different models)")
print(f"  order-inconsistent: {n_oi}/300 = {n_oi / 300:.3f}  (of {n_complete} complete items: {n_oi / max(n_complete, 1):.3f})")
print(f"  items with a TIE in one or both orders (reported separately, not order-inconsistent): {n_tie_item}")
out["order_inconsistent"] = {"n": n_oi, "share_of_300": n_oi / 300, "tie_items": n_tie_item, "complete": n_complete}

# item 4: consolidated per (pair, prompt)
print("\n[4] Consolidated outcome per (pair, prompt)")
print(f"  consistent: {sum(1 for k, c in cat.items() if len(k) == 2 and c == 'consistent')}  "
      f"order_inconsistent: {n_oi}  TIE-involved: {n_tie_item}  (total complete {n_complete})")
print("  per pair: model a wins / model b wins / half-wins (order-inconsistent or TIE), out of complete items")
cons_table = []
for pi, (a, b) in enumerate(PAIRS):
    ca = sum(1 for pid in PIDS if cat.get((pi, pid)) == "consistent" and S[pi, pid - 1] == 1.0)
    cb = sum(1 for pid in PIDS if cat.get((pi, pid)) == "consistent" and S[pi, pid - 1] == 0.0)
    h = sum(1 for pid in PIDS if cat.get((pi, pid)) in ("order_inconsistent", "TIE"))
    nc = int(V[pi].sum())
    cons_table.append({"pair": f"{a} vs {b}", "a_wins": ca, "b_wins": cb, "half": h, "n": nc,
                       "a_score": float(S[pi].sum()), "b_score": float(nc - S[pi].sum())})
    print(f"  {a} vs {b}: {ca} / {cb} / {h}   (n={nc}; score {a}={S[pi].sum():.1f}, {b}={nc - S[pi].sum():.1f})")
out["consolidated"] = cons_table

# main BT and sensitivity
print("\n[5] Bradley-Terry (consolidated), Elo scale, ts_1M = 1000")
print(f"  fit used: {method_main}; bootstrap fits: unregularized={boot_methods['unregularized MLE']}, ridge={boot_methods['ridge']}")
print(f"  {'model':<10}{'Elo':>9}{'95% CI (prompt bootstrap)':>34}")
for m in MODELS:
    lo_, hi_ = elo_ci[m]
    print(f"  {m:<10}{elo_main[m]:>9.1f}{f'[{lo_:.1f}, {hi_:.1f}]':>34}")
out["bt_main"] = {"method": method_main, "elo": elo_main, "ci": elo_ci,
                  "bootstrap_fit_methods": boot_methods}

print("\n[6] Sensitivity: BT on all raw order-level verdicts with slot-A (order) parameter, no consolidation")
print(f"  status: {st_raw}; n verdicts: {len(raw_recs)}; order (slot-A) advantage gamma = {gamma:.3f} "
      f"(P(slot A wins | equal ratings) = {float(sig(gamma)):.3f})")
print(f"  {'model':<10}{'Elo (raw+order)':>17}{'Elo main (consol.)':>22}")
for m in MODELS:
    print(f"  {m:<10}{elo_raw[m]:>17.1f}{elo_main[m]:>22.1f}")
out["sensitivity"] = {"status": st_raw, "gamma": float(gamma), "elo": elo_raw, "n": len(raw_recs)}

# item 7: win-rate matrix (order-level verdicts, TIE = 0.5)
print("\n[7] Win-rate matrix: row model's win rate vs column model (order-level verdicts, TIE = 0.5)")
aw = np.zeros((len(PAIRS), 50)); bw = np.zeros((len(PAIRS), 50)); tt = np.zeros((len(PAIRS), 50)); nn = np.zeros((len(PAIRS), 50))
wr_rows = []
for pi, (a, b) in enumerate(PAIRS):
    for pid in PIDS:
        for order in ("ab", "ba"):
            v = verdict_by.get((a, b, pid, order))
            if v is None:
                continue
            wnr = winner_of(v, a, b, order)
            nn[pi, pid - 1] += 1
            if wnr == "TIE":
                tt[pi, pid - 1] += 1
            elif wnr == a:
                aw[pi, pid - 1] += 1
            else:
                bw[pi, pid - 1] += 1
    na = int(nn[pi].sum()); ka = int(aw[pi].sum()); kb = int(bw[pi].sum()); kt = int(tt[pi].sum())
    rate = (ka + 0.5 * kt) / na if na else float("nan")
    num_b = C @ (aw[pi] + 0.5 * tt[pi]); den_b = C @ nn[pi]
    bs = num_b / np.where(den_b > 0, den_b, np.nan)
    lo_, hi_ = float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    wr_rows.append({"pair": f"{a} vs {b}", "a_wins": ka, "b_wins": kb, "ties": kt, "n": na,
                    "rate_a": rate, "ci": [lo_, hi_]})
    print(f"  {a} vs {b}: {a} {ka} / {b} {kb} / TIE {kt}  (n={na});  rate({a})={rate:.3f} "
          f"[{lo_:.3f}, {hi_:.3f}]  rate({b})={1 - rate:.3f}")
out["win_matrix"] = wr_rows

# item 8: implied vs observed (consolidated)
print("\n[8] Implied win probability from ratings vs observed (consolidated) win rate, all 6 pairs")
print(f"  {'pair (row vs col)':<26}{'observed':>10}{'implied':>10}{'diff':>9}  flag(>0.10)")
imp_rows = []
for pi, (a, b) in enumerate(PAIRS):
    nc = V[pi].sum()
    obs = float(S[pi].sum() / nc) if nc else float("nan")
    ia, ib = MODELS.index(a), MODELS.index(b)
    imp = float(sig(th_main[ia] - th_main[ib]))
    imp_elo = 1 / (1 + 10 ** ((elo_main[b] - elo_main[a]) / 400))
    assert abs(imp - imp_elo) < 1e-9
    d = imp - obs
    flag = "FLAG" if abs(d) > 0.10 else ""
    print(f"  {a + ' vs ' + b:<26}{obs:>10.3f}{imp:>10.3f}{d:>+9.3f}  {flag}")
    imp_rows.append({"pair": f"{a} vs {b}", "observed": obs, "implied": imp, "diff": d, "flag": flag})
flagged = [r["pair"] for r in imp_rows if r["flag"]]
print(f"  flagged pairs: {flagged if flagged else 'none'}")
print(f"  scale check: implied via Elo formula equals sigmoid(theta_a - theta_b) for all pairs (asserted).")
print(f"  regularization in main fit: {'yes (ridge %s)' % RIDGE if method_main.startswith('ridge') else 'none'}")
out["implied_vs_observed"] = imp_rows

# item 9: length vs outcome
ntok = {}
with (ANALYSE / "generations.csv").open(encoding="utf-8", newline="") as f:
    csv.field_size_limit(10**9)
    for r in csv.DictReader(f):
        if int(r["completion_idx"]) == 0:
            ntok[(r["model"], int(r["prompt_id"]))] = int(r["n_tokens"])
xs, ys, pidx = [], [], []
for pi, (a, b) in enumerate(PAIRS):
    for pid in PIDS:
        if V[pi, pid - 1]:
            xs.append(ntok[(a, pid)] - ntok[(b, pid)])
            ys.append(2 * S[pi, pid - 1] - 1)  # +1 a consistent win, -1 b consistent win, 0 half
            pidx.append(pid - 1)
xs = np.asarray(xs, float); ys = np.asarray(ys, float); pidx = np.asarray(pidx)


def wpearson(x, y, w):
    sw = w.sum()
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    cov = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum()
    vy = (w * (y - my) ** 2).sum()
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


r_obs = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
rb = []
for r in range(BOOT_N):
    w = C[r][pidx]
    if w.sum() == 0:
        continue
    rb.append(wpearson(xs, ys, w))
rb = np.asarray(rb)
r_lo, r_hi = float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))
print("\n[9] Length vs outcome")
print(f"  Pearson r(token-length difference a-b, outcome +1/0/-1) over {len(xs)} items = {r_obs:.3f}  "
      f"95% CI (prompt bootstrap) [{r_lo:.3f}, {r_hi:.3f}]")
out["length"] = {"r": r_obs, "ci": [r_lo, r_hi], "n": len(xs)}

# item 10: agreement with Ministral
print("\n[10] Agreement with Ministral-14B on the same jobs")
both = []
for j in jobs:
    k = j["key"]
    vm = min_v.get(k)
    vs = verdict_of[k]
    if vm is not None and vs is not None:
        both.append((vm, vs))
dec = [(m, s) for m, s in both if m in ("A", "B") and s in ("A", "B")]
n_dec = len(dec)
po = sum(1 for m, s in dec if m == s) / n_dec
pm = {l: sum(1 for m, _ in dec if m == l) / n_dec for l in ("A", "B")}
ps = {l: sum(1 for _, s in dec if s == l) / n_dec for l in ("A", "B")}
pe = sum(pm[l] * ps[l] for l in ("A", "B"))
kappa = (po - pe) / (1 - pe)
print(f"  jobs with verdicts from both judges: {len(both)} (Ministral missing: "
      f"{sum(1 for j in jobs if min_v.get(j['key']) is None)}, second judge missing: "
      f"{sum(1 for j in jobs if verdict_of[j['key']] is None)})")
print(f"  Cohen's kappa on decisive verdicts (both A/B): {kappa:.3f}  (n={n_dec}, raw agreement {po:.3f})")
mins_consistent = sum(1 for (pi, pid), c in cat_m.items() if c == "consistent")
mins_complete = int(V_m.sum())
sec_consistent = sum(1 for k, c in cat.items() if len(k) == 2 and c == "consistent")
print(f"  share of complete pairs with the same winner in both orders: "
      f"second judge {sec_consistent}/{n_complete} = {sec_consistent / max(n_complete, 1):.3f}; "
      f"Ministral {mins_consistent}/{mins_complete} = {mins_consistent / max(mins_complete, 1):.3f}")
print(f"  Ministral BT fit: {method_m}")
print(f"  {'pair (first vs second)':<24}{'second judge: rate(first) / Elo diff':>40}{'Ministral: rate(first) / Elo diff':>40}")
order_agree = {}
for a, b in [("ours_33M", "ts_33M"), ("ours_1M", "ts_1M")]:
    pi = PAIRS.index((a, b))
    sa = float(S[pi].sum() / max(V[pi].sum(), 1))
    sm = float(S_m[pi].sum() / max(V_m[pi].sum(), 1))
    d2 = float(elo_main[a] - elo_main[b])
    dm = float(elo_m[a] - elo_m[b])
    order_agree[f"{a} vs {b}"] = {"second_judge_rate_a": sa, "ministral_rate_a": sm,
                                  "second_judge_elo_diff": d2, "ministral_elo_diff": dm,
                                  "same_direction_rate": bool((sa > 0.5) == (sm > 0.5)),
                                  "same_direction_elo": bool((d2 > 0) == (dm > 0))}
    print(f"  {a + ' vs ' + b:<24}{f'{sa:.3f} / {d2:+.1f}':>40}{f'{sm:.3f} / {dm:+.1f}':>40}")
    print(f"    same ordering by win rate: {(sa > 0.5) == (sm > 0.5)};  same ordering by Elo: {(d2 > 0) == (dm > 0)}")
print("  Elo (Ministral BT):", {m: round(float(elo_m[m]), 1) for m in MODELS})
out["agreement"] = {"kappa_decisive": kappa, "n_decisive": n_dec, "raw_agreement": po,
                    "consistent_share_second": sec_consistent / max(n_complete, 1),
                    "consistent_share_ministral": mins_consistent / max(mins_complete, 1),
                    "ordering": order_agree, "ministral_elo": elo_m, "ministral_method": method_m}

with (HERE / "analysis.json").open("w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print("\nwrote verdicts.csv, failures_log.txt, analysis.json")
