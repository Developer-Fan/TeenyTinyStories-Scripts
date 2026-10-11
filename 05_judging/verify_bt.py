"""Independent check of analyze.py: rebuild outcomes from verdicts.csv and fit BT with scipy.optimize."""
import csv, json, math, re
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
MODELS = ["ours_1M","ours_33M","ts_1M","ts_33M"]
rows = list(csv.DictReader(open("verdicts.csv", encoding="utf-8", newline="")))
assert len(rows) == 600, len(rows)
v = {}
for r in rows:
    a, b = r["pair"].split(" vs ")
    v[(a, b, int(r["prompt_id"]), r["order"])] = r["parsed_verdict"]
pairs = [(a,b) for i,a in enumerate(MODELS) for b in MODELS[i+1:]]
def wnr(x, a, b, o):
    if x == "TIE": return "TIE"
    A, B = (a, b) if o == "ab" else (b, a)
    return A if x == "A" else B
items = []  # (i, j, s_a)
raw = []    # (X=slotA idx, Y=slotB idx, s)
for a, b in pairs:
    for pid in range(1, 51):
        x1, x2 = v[(a,b,pid,"ab")], v[(a,b,pid,"ba")]
        w1, w2 = wnr(x1,a,b,"ab"), wnr(x2,a,b,"ba")
        if "TIE" in (w1, w2): s = 0.5
        elif w1 == w2: s = 1.0 if w1 == a else 0.0
        else: s = 0.5
        items.append((MODELS.index(a), MODELS.index(b), s))
for r in rows:
    a, b = r["pair"].split(" vs ")
    A, B = (a, b) if r["order"] == "ab" else (b, a)
    s = 1.0 if r["parsed_verdict"] == "A" else (0.0 if r["parsed_verdict"] == "B" else 0.5)
    raw.append((MODELS.index(A), MODELS.index(B), s))
# consolidated BT, ts_1M fixed at 0
def nll(q):
    th = np.array([q[0], q[1], 0.0, q[2]])
    tot = 0.0
    for i, j, s in items:
        z = th[i] - th[j]
        tot -= s * (-np.logaddexp(0, -z)) + (1 - s) * (-np.logaddexp(0, z))
    return tot
res = minimize(nll, np.zeros(3), method="BFGS", options={"gtol": 1e-10})
th = np.array([res.x[0], res.x[1], 0.0, res.x[2]])
elo = {m: 1000 + 400/math.log(10)*(th[i]-th[2]) for i,m in enumerate(MODELS)}
print("scipy consolidated BT Elo:", {m: round(e,1) for m,e in elo.items()}, "converged:", res.success)
# raw + order parameter, ts_1M fixed
def nll_raw(q):
    th = np.array([q[0], q[1], 0.0, q[2]]); g = q[3]
    tot = 0.0
    for X, Y, s in raw:
        z = th[X] - th[Y] + g
        tot -= s * (-np.logaddexp(0, -z)) + (1 - s) * (-np.logaddexp(0, z))
    return tot
res2 = minimize(nll_raw, np.zeros(4), method="BFGS", options={"gtol": 1e-10})
th2 = np.array([res2.x[0], res2.x[1], 0.0, res2.x[2]])
elo2 = {m: 1000 + 400/math.log(10)*(th2[i]-th2[2]) for i,m in enumerate(MODELS)}
print("scipy raw+order BT Elo:", {m: round(e,1) for m,e in elo2.items()}, "gamma:", round(res2.x[3],3), "converged:", res2.success)
an = json.load(open("analysis.json"))
print("analyze.py consolidated Elo:", {m: round(e,1) for m,e in an["bt_main"]["elo"].items()})
print("analyze.py raw+order Elo:   ", {m: round(e,1) for m,e in an["sensitivity"]["elo"].items()}, "gamma:", round(an["sensitivity"]["gamma"],3))
mx = max(abs(elo[m]-an["bt_main"]["elo"][m]) for m in MODELS)
mx2 = max(abs(elo2[m]-an["sensitivity"]["elo"][m]) for m in MODELS)
print(f"max |diff| consolidated = {mx:.4f} Elo; raw = {mx2:.4f} Elo")
# verdict-text check: any response mentioning TIE as a verdict in a form the regex could miss?
recs = [json.loads(l) for l in open("calls_log.jsonl", encoding="utf-8")]
tie_like = [r["key"] for r in recs if r["response_text"] and re.search(r"verdict[^\n]{0,20}tie", r["response_text"], re.I)]
print("responses containing 'verdict ... tie' anywhere:", len(tie_like), "| parsed TIE count in verdicts.csv:", sum(1 for r in rows if r["parsed_verdict"]=="TIE"))
print("verdicts.csv rows:", len(rows), "| parse statuses:", set(r["parse_status_by_attempt"] for r in rows))
