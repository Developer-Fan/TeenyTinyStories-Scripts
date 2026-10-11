#!/usr/bin/env python3
"""Step 4 export + Step 5 analysis for the EASY-prompt Luna judge run.

Mirrors second_judge_out/analyze.py with:
  PIDS 201-250 (easy prompts, T=0.8 generations, ci0)
  lengths from easy2000_gens.jsonl (ci0 n_tokens)
  item 10 replaced: agreement Luna-easy vs my blind-100 (matched pairs, ci0),
  plus Elo comparison vs the main hard-prompt Luna run.
Reads (read-only): jobs_600.jsonl, calls_log.jsonl (this dir);
  easy2000_gens.jsonl; blind100.json; second_judge_out/analysis.json.
Writes: verdicts.csv, failures_log.txt, missing_verdicts.txt, analysis.json.
"""
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SECOND = Path("/Users/yanfu/Documents/TeenyTinyStories/analyse/second_judge_out")
EASY_GENS = Path("/var/folders/z8/4b7hbhhn259d_hvt7zl7cw340000gp/T/opencode/easy2000_gens.jsonl")
BLIND100 = Path("/var/folders/z8/4b7hbhhn259d_hvt7zl7cw340000gp/T/opencode/blind100.json")
MODEL_REQ = "infron:openai/gpt-5.6-luna"

MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
PAIRS = list(itertools.combinations(MODELS, 2))
PIDS = list(range(201, 251))
PID0 = 201
ANCHOR = "ts_1M"
AI = MODELS.index(ANCHOR)
ELO_SCALE = 400 / math.log(10)
RIDGE = 0.01
BOOT_N = 2000
BOOT_SEED = 0
Z = 1.959963984540054
RX = re.compile(r"^\s*Verdict:\s*(A|B|TIE)\s*$", re.IGNORECASE | re.MULTILINE)

# my blind-100 verdicts by pair number (A/B/TIE = slot shown to me)
ME_A = {1, 4, 5, 7, 8, 9, 10, 12, 13, 14, 18, 19, 20, 21, 27, 28, 30, 35, 37, 38,
        39, 46, 47, 48, 49, 54, 56, 61, 62, 67, 68, 71, 73, 76, 78, 79, 83, 84, 85,
        86, 87, 90, 92, 93, 95, 100}
ME_B = {3, 6, 11, 17, 22, 23, 24, 25, 26, 29, 32, 33, 34, 36, 40, 41, 42, 44, 45,
        50, 51, 52, 53, 55, 57, 58, 59, 60, 63, 64, 66, 69, 70, 72, 74, 75, 77, 80,
        81, 82, 88, 89, 91, 94, 96, 98}
ME_TIE = {2, 15, 16, 31, 43, 65, 97, 99}
assert len(ME_A) + len(ME_B) + len(ME_TIE) == 100


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


def winner_of(v, a, b, order):
    if v is None:
        return None
    if v == "TIE":
        return "TIE"
    slotA, slotB = (a, b) if order == "ab" else (b, a)
    return slotA if v == "A" else slotB


resolved = {}
failures = []
retry_used = []
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

# ---------------------------------------------------------------- consolidation (positional cols 0..49)
verdict_by = {}
for j in jobs:
    t = j["tag"]
    verdict_by[(t["a"], t["b"], t["pid"], t["order"])] = verdict_of[j["key"]]

S = np.zeros((len(PAIRS), 50))
V = np.zeros((len(PAIRS), 50))
cat = {}


def col(pid):
    return pid - PID0


for pi, (a, b) in enumerate(PAIRS):
    for pid in PIDS:
        vab = verdict_by.get((a, b, pid, "ab"))
        vba = verdict_by.get((a, b, pid, "ba"))
        if vab is None or vba is None:
            continue
        wab = winner_of(vab, a, b, "ab")
        wba = winner_of(vba, a, b, "ba")
        if wab == "TIE" or wba == "TIE":
            s = 0.5
            cat[(pi, pid)] = "TIE"
        elif wab == wba:
            s = 1.0 if wab == a else 0.0
            cat[(pi, pid)] = "consistent"
        else:
            s = 0.5
            cat[(pi, pid)] = "order_inconsistent"
        S[pi, col(pid)] = s
        V[pi, col(pid)] = 1.0
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

rng = np.random.default_rng(BOOT_SEED)
idx = rng.integers(0, 50, size=(BOOT_N, 50))
C = np.zeros((BOOT_N, 50))
np.add.at(C, (np.repeat(np.arange(BOOT_N), 50), idx.ravel()), 1)

Wb = C @ (S * V).T
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


def fit_raw(recs):
    free = [0, 1, 3]
    q = np.zeros(4)
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

# ---------------------------------------------------------------- output
out = {}
print("=" * 78)
print("EASY-JUDGE RUN (T=0.8 gens, easy prompts 201-250): analysis output")
print("=" * 78)
print(f"model requested: {MODEL_REQ}")
print(f"jobs: {len(jobs)}  valid verdicts: {n_valid}/{len(jobs)}  failed jobs: {len(jobs) - n_valid}")
print(f"complete (pair,prompt) items: {n_complete}/300")
missing = [j for j in jobs if verdict_of[j["key"]] is None]
with (HERE / "missing_verdicts.txt").open("w", encoding="utf-8") as f:
    f.write("prompt_id\tpair\torder\tkey\tattempts_logged\n")
    for j in missing:
        t = j["tag"]
        f.write(f"{t['pid']}\t{t['a']} vs {t['b']}\t{t['order']}\t{j['key']}\t{n_att[j['key']]}\n")
print(f"missing verdicts: {len(missing)}")
missing = [j["key"] for j in missing]
out["counts"] = {"jobs": len(jobs), "valid": n_valid, "failed_jobs": missing,
                 "failed_attempts": len(failures), "retries": len(retry_used),
                 "complete_items": n_complete}

valid_v = [v for v in verdict_of.values() if v is not None]
nB = sum(1 for v in valid_v if v == "B")
nA = sum(1 for v in valid_v if v == "A")
nT = sum(1 for v in valid_v if v == "TIE")
n_all = len(valid_v)
lo, hi = wilson(nB, n_all)
lod, hid = wilson(nB, nA + nB)
print("\n[1] Position bias (second-listed = slot B)")
print(f"  second-listed won: {nB}/{n_all} = {nB / n_all:.3f}  95% Wilson CI [{lo:.3f}, {hi:.3f}]")
out["position"] = {"B": nB, "A": nA, "TIE": nT, "n": n_all, "B_share": nB / n_all, "B_ci": [lo, hi]}

print("\n[2] Direct TIE verdicts")
print(f"  TIE: {nT}/{n_all} = {nT / n_all:.3f}")
out["ties"] = {"TIE": nT, "n": n_all}

n_oi = sum(1 for k, c in cat.items() if len(k) == 2 and c == "order_inconsistent")
n_tie_item = sum(1 for k, c in cat.items() if len(k) == 2 and c == "TIE")
print("\n[3] Order-inconsistent items")
print(f"  order-inconsistent: {n_oi}/{n_complete}")
out["order_inconsistent"] = {"n": n_oi, "tie_items": n_tie_item, "complete": n_complete}

print("\n[4] Consolidated outcome per (pair, prompt)")
cons_table = []
for pi, (a, b) in enumerate(PAIRS):
    ca = sum(1 for pid in PIDS if cat.get((pi, pid)) == "consistent" and S[pi, col(pid)] == 1.0)
    cb = sum(1 for pid in PIDS if cat.get((pi, pid)) == "consistent" and S[pi, col(pid)] == 0.0)
    h = sum(1 for pid in PIDS if cat.get((pi, pid)) in ("order_inconsistent", "TIE"))
    nc = int(V[pi].sum())
    cons_table.append({"pair": f"{a} vs {b}", "a_wins": ca, "b_wins": cb, "half": h, "n": nc,
                       "a_score": float(S[pi].sum()), "b_score": float(nc - S[pi].sum())})
    print(f"  {a} vs {b}: {ca} / {cb} / {h}   (n={nc}; score {a}={S[pi].sum():.1f}, {b}={nc - S[pi].sum():.1f})")
out["consolidated"] = cons_table

print("\n[5] Bradley-Terry (consolidated), Elo scale, ts_1M = 1000")
print(f"  fit used: {method_main}; bootstrap fits: unregularized={boot_methods['unregularized MLE']}, ridge={boot_methods['ridge']}")
for m in MODELS:
    lo_, hi_ = elo_ci[m]
    print(f"  {m:<10}{elo_main[m]:>9.1f}{f'[{lo_:.1f}, {hi_:.1f}]':>34}")
out["bt_main"] = {"method": method_main, "elo": elo_main, "ci": elo_ci,
                  "bootstrap_fit_methods": boot_methods}

print("\n[6] Sensitivity: raw + order parameter")
print(f"  status: {st_raw}; gamma = {gamma:.3f} (P(slot A wins | equal) = {float(sig(gamma)):.3f})")
for m in MODELS:
    print(f"  {m:<10}{elo_raw[m]:>17.1f}{elo_main[m]:>22.1f}")
out["sensitivity"] = {"status": st_raw, "gamma": float(gamma), "elo": elo_raw, "n": len(raw_recs)}

print("\n[7] Win-rate matrix (order-level)")
wr_rows = []
for pi, (a, b) in enumerate(PAIRS):
    ka = kb = kt = na = 0
    for pid in PIDS:
        for order in ("ab", "ba"):
            v = verdict_by.get((a, b, pid, order))
            if v is None:
                continue
            wnr = winner_of(v, a, b, order)
            na += 1
            if wnr == "TIE":
                kt += 1
            elif wnr == a:
                ka += 1
            else:
                kb += 1
    rate = (ka + 0.5 * kt) / na if na else float("nan")
    # prompt-bootstrap CI via positional resample
    aw = np.zeros(50)
    bw = np.zeros(50)
    tt = np.zeros(50)
    nn = np.zeros(50)
    for pid in PIDS:
        for order in ("ab", "ba"):
            v = verdict_by.get((a, b, pid, order))
            if v is None:
                continue
            wnr = winner_of(v, a, b, order)
            nn[col(pid)] += 1
            if wnr == "TIE":
                tt[col(pid)] += 1
            elif wnr == a:
                aw[col(pid)] += 1
            else:
                bw[col(pid)] += 1
    num_b = C @ (aw + 0.5 * tt)
    den_b = C @ nn
    bs = num_b / np.where(den_b > 0, den_b, np.nan)
    lo_, hi_ = float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    wr_rows.append({"pair": f"{a} vs {b}", "a_wins": ka, "b_wins": kb, "ties": kt, "n": na,
                    "rate_a": rate, "ci": [lo_, hi_]})
    print(f"  {a} vs {b}: {a} {ka} / {b} {kb} / TIE {kt}  (n={na});  rate({a})={rate:.3f} [{lo_:.3f}, {hi_:.3f}]")
out["win_matrix"] = wr_rows

print("\n[8] Implied vs observed (consolidated)")
imp_rows = []
for pi, (a, b) in enumerate(PAIRS):
    nc = V[pi].sum()
    obs = float(S[pi].sum() / nc) if nc else float("nan")
    ia, ib = MODELS.index(a), MODELS.index(b)
    imp = float(sig(th_main[ia] - th_main[ib]))
    d = imp - obs
    flag = "FLAG" if abs(d) > 0.10 else ""
    print(f"  {a + ' vs ' + b:<26}{obs:>10.3f}{imp:>10.3f}{d:>+9.3f}  {flag}")
    imp_rows.append({"pair": f"{a} vs {b}", "observed": obs, "implied": imp, "diff": d, "flag": flag})
out["implied_vs_observed"] = imp_rows

# item 9: length vs outcome (easy2000 ci0 n_tokens)
ntok = {}
for line in EASY_GENS.open(encoding="utf-8"):
    r = json.loads(line)
    if r["completion_idx"] == 0:
        ntok[(r["model"], int(r["prompt_id"]))] = int(r["n_tokens"])
xs, ys, pidx = [], [], []
for pi, (a, b) in enumerate(PAIRS):
    for pid in PIDS:
        if V[pi, col(pid)]:
            xs.append(ntok[(a, pid)] - ntok[(b, pid)])
            ys.append(2 * S[pi, col(pid)] - 1)
            pidx.append(col(pid))
xs = np.asarray(xs, float)
ys = np.asarray(ys, float)
pidx = np.asarray(pidx)


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
print(f"  Pearson r = {r_obs:.3f}  95% CI [{r_lo:.3f}, {r_hi:.3f}]  n={len(xs)}")
out["length"] = {"r": r_obs, "ci": [r_lo, r_hi], "n": len(xs)}

# item 10: agreement Luna-easy vs me (100 matched-pair ci0 items) + vs hard run
blind = json.load(BLIND100.open(encoding="utf-8"))
me_win = {}
for p in blind:
    n = p["n"]
    if n in ME_A:
        me_win[n] = p["A_is"]
    elif n in ME_B:
        me_win[n] = p["B_is"]
    else:
        me_win[n] = "tie"
blind_by_pair_pid = {}
for p in blind:
    # recover pair + pid from jobs: match on prompt text
    blind_by_pair_pid[p["n"]] = p
# map blind n -> (pair, pid): pairs were built per pid over [1M,33M] in pid order
agree_items = []
for p in blind:
    n = p["n"]
    # reconstruct: pids 201..250, two pairs each in order
    idx0 = n - 1
    pid = 201 + idx0 // 2
    pair = ("ours_1M", "ts_1M") if idx0 % 2 == 0 else ("ours_33M", "ts_33M")
    a, b = pair
    va = verdict_by.get((a, b, pid, "ab"))
    vb = verdict_by.get((a, b, pid, "ba"))
    if va is None or vb is None:
        continue
    wa = winner_of(va, a, b, "ab")
    wb = winner_of(vb, a, b, "ba")
    luna = wa if wa == wb else "tie-half"
    if luna == "TIE":
        luna = "tie-half"
    mw = me_win[n]
    ml = "tie" if mw == "tie" else ("ours" if mw.startswith("ours") else "ts")
    ll = "tie" if luna in ("tie-half", "TIE") else ("ours" if luna.startswith("ours") else "ts")
    agree_items.append((ml, ll, pair))
po = sum(1 for h, l, _ in agree_items if h == l) / len(agree_items)
labels = ["ours", "tie", "ts"]
pe = sum((sum(1 for h, _, _ in agree_items if h == L) / len(agree_items)) *
         (sum(1 for _, l, _ in agree_items if l == L) / len(agree_items)) for L in labels)
kappa = (po - pe) / (1 - pe)
dec = [(h, l) for h, l, _ in agree_items if h != "tie" and l != "tie"]
po_d = sum(1 for h, l in dec if h == l) / len(dec) if dec else float("nan")
print("\n[10] Agreement Luna-easy vs me (100 matched-pair ci0 items)")
print(f"  exact (ours/tie/ts): {po:.3f}  kappa={kappa:.3f}  n={len(agree_items)}")
print(f"  decisive-only: {po_d:.3f}  n={len(dec)}")
for pair in [("ours_1M", "ts_1M"), ("ours_33M", "ts_33M")]:
    sub = [(h, l) for h, l, pp in agree_items if pp == pair]
    pod = sum(1 for h, l in sub if h == l) / len(sub)
    print(f"  {pair[0]} vs {pair[1]}: exact {pod:.3f} n={len(sub)}")
out["agreement_me"] = {"po": po, "kappa": kappa, "n": len(agree_items),
                       "decisive_po": po_d, "decisive_n": len(dec)}

hard = json.load((SECOND / "analysis.json").open(encoding="utf-8"))
print("\n[11] Luna-easy vs Luna-hard (main run) Elo")
print(f"  {'model':<10}{'easy':>9}{'hard':>9}")
for m in MODELS:
    print(f"  {m:<10}{elo_main[m]:>9.1f}{hard['bt_main']['elo'][m]:>9.1f}")
out["vs_hard"] = {"easy_elo": elo_main, "hard_elo": hard["bt_main"]["elo"],
                  "hard_ci": hard["bt_main"]["ci"]}

with (HERE / "analysis.json").open("w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print("\nwrote verdicts.csv, failures_log.txt, missing_verdicts.txt, analysis.json")
