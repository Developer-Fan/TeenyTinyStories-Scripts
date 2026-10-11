"""Build the 600 pairwise attempt inputs for the second-judge run.

Template constants are imported unmodified from the Ministral-run script
cloud_training/eval/build_jobs.py. Inputs are read-only. Writes only into this directory:
  jobs_600.jsonl        600 pair/prompt/order jobs (keys identical to the Ministral run)
  pilot_selection.json  seed 0 sample of 20 job keys for the Step 3 parse pilot
  pilot_attempt1.jsonl  attempt-1 inputs for the 20 pilot jobs
"""
import csv
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSE = Path("/Users/yanfu/Documents/TeenyTinyStories/analyse")
EVAL = Path("/Users/yanfu/Documents/TeenyTinyStories/cloud_training/eval")
sys.path.insert(0, str(EVAL))
import build_jobs as B  # noqa: E402  (template constants only; module has no side effects)
from prompts_lib import load_prompts  # noqa: E402

PROMPTS = ANALYSE / "prompts_official_style.txt"
GENS = ANALYSE / "generations.csv"
MIN_JOBS = EVAL / "eval_outputs" / "jobs_pairwise.jsonl"
EXPECTED_PROMPT_SHA = "005759a4cc99efd7ead3460704743b2d056e1ef66f53d1bc0d7c2b04e871d81b"
MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
PILOT_SEED = 0
PILOT_N = 20

# --- prompts (same parsing rules as prompts_lib.load_prompts) ---
raw = PROMPTS.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_PROMPT_SHA, "prompt hash mismatch"
P = {}
for line in raw.decode("utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    m = re.match(r"^(\d+)\.\s*(.*)$", line)
    pid, text = int(m.group(1)), m.group(2)
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'").rstrip())
    P[pid] = text
assert sorted(P) == list(range(1, 51))
assert P == load_prompts(), "analyse/ prompts differ from the copy the Ministral run loaded"

# --- completions (completion_idx 0 only is used by the protocol) ---
csv.field_size_limit(10**9)
G = {}
with GENS.open(encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        G[(r["model"], int(r["prompt_id"]), int(r["completion_idx"]))] = r["text"]
assert len(G) == 2000

# --- 600 jobs ---
jobs = []
for a, b in itertools.combinations(MODELS, 2):
    for pid in range(1, 51):
        ta, tb = G[(a, pid, 0)], G[(b, pid, 0)]
        for order, (x, y, xn, yn) in (("ab", (ta, tb, a, b)), ("ba", (tb, ta, b, a))):
            body = (f"{B.PAIR_HEAD}{P[pid]}{B.PAIR_MID_A}{P[pid]}***{x}"
                    f"{B.PAIR_MID_B}{P[pid]}***{y}{B.PAIR_TAIL}")
            jobs.append({"key": f"p|{a}|{b}|{pid}|{order}",
                         "messages": [{"role": "user", "content": body}],
                         "tag": {"phase": "pair", "a": a, "b": b, "pid": pid, "order": order,
                                 "A_is": xn, "B_is": yn}})
assert len(jobs) == 600 and len({j["key"] for j in jobs}) == 600

# --- cross-check against the Ministral job file (read-only) ---
min_by_key = {}
with MIN_JOBS.open(encoding="utf-8") as f:
    for l in f:
        j = json.loads(l)
        min_by_key[j["key"]] = j
assert set(min_by_key) == {j["key"] for j in jobs}, "key sets differ"
bad_msg = [j["key"] for j in jobs if min_by_key[j["key"]]["messages"] != j["messages"]]
bad_tag = [j["key"] for j in jobs if min_by_key[j["key"]]["tag"] != j["tag"]]
assert not bad_msg and not bad_tag, (bad_msg[:3], bad_tag[:3])

with (HERE / "jobs_600.jsonl").open("w", encoding="utf-8") as f:
    for j in jobs:
        f.write(json.dumps(j, ensure_ascii=False) + "\n")

# --- Step 3 pilot selection: seed 0 over the sorted key list ---
keys = sorted(j["key"] for j in jobs)
pilot_keys = random.Random(PILOT_SEED).sample(keys, PILOT_N)
with (HERE / "pilot_selection.json").open("w", encoding="utf-8") as f:
    json.dump({"seed": PILOT_SEED,
               "method": "random.Random(0).sample(sorted list of 600 job keys, 20)",
               "n": PILOT_N, "keys": pilot_keys}, f, indent=2)
byk = {j["key"]: j for j in jobs}
with (HERE / "pilot_attempt1.jsonl").open("w", encoding="utf-8") as f:
    for k in pilot_keys:
        j = byk[k]
        f.write(json.dumps({"key": k, "attempt": 1, "phase": "pilot", "tag": j["tag"],
                            "messages": j["messages"]}, ensure_ascii=False) + "\n")

print("jobs_600: 600 (messages and tags identical to Ministral jobs_pairwise.jsonl)")
print("pilot keys:")
for k in pilot_keys:
    print("  ", k)
