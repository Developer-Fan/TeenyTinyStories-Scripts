"""Build the 600 pairwise attempt inputs for the easy-prompt Luna judge run.

Same template constants as the main run (cloud_training/eval/build_jobs.py).
Completions: ci0 from easy2000_gens.jsonl (T=0.8). Prompts: ids 201-250.
Writes only into analyse/easy_judge/:
  jobs_600.jsonl, pilot_selection.json, pilot_attempt1.jsonl
"""
import itertools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = Path("/Users/yanfu/Documents/TeenyTinyStories/cloud_training/eval")
EASY_GENS = Path("/var/folders/z8/4b7hbhhn259d_hvt7zl7cw340000gp/T/opencode/easy2000_gens.jsonl")
sys.path.insert(0, str(EVAL))
import build_jobs as B  # noqa: E402 template constants only

MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
PILOT_SEED = 0
PILOT_N = 20

G, P = {}, {}
for line in EASY_GENS.open(encoding="utf-8"):
    r = json.loads(line)
    if r["completion_idx"] == 0:
        G[(r["model"], int(r["prompt_id"]))] = r["text"]
        P[int(r["prompt_id"])] = r["prompt"]
assert sorted(P) == list(range(201, 251)), sorted(P)[:5]
assert len(G) == 200, len(G)

jobs = []
for a, b in itertools.combinations(MODELS, 2):
    for pid in range(201, 251):
        ta, tb = G[(a, pid)], G[(b, pid)]
        for order, (x, y, xn, yn) in (("ab", (ta, tb, a, b)), ("ba", (tb, ta, b, a))):
            body = (f"{B.PAIR_HEAD}{P[pid]}{B.PAIR_MID_A}{P[pid]}***{x}"
                    f"{B.PAIR_MID_B}{P[pid]}***{y}{B.PAIR_TAIL}")
            jobs.append({"key": f"e|{a}|{b}|{pid}|{order}",
                         "messages": [{"role": "user", "content": body}],
                         "tag": {"phase": "pair", "a": a, "b": b, "pid": pid, "order": order,
                                 "A_is": xn, "B_is": yn}})
assert len(jobs) == 600 and len({j["key"] for j in jobs}) == 600

with (HERE / "jobs_600.jsonl").open("w", encoding="utf-8") as f:
    for j in jobs:
        f.write(json.dumps(j, ensure_ascii=False) + "\n")

keys = sorted(j["key"] for j in jobs)
pilot_keys = random.Random(PILOT_SEED).sample(keys, PILOT_N)
with (HERE / "pilot_selection.json").open("w", encoding="utf-8") as f:
    json.dump({"seed": PILOT_SEED,
               "method": "random.Random(0).sample(sorted list of 600 easy job keys, 20)",
               "n": PILOT_N, "keys": pilot_keys}, f, indent=2)
byk = {j["key"]: j for j in jobs}
with (HERE / "pilot_attempt1.jsonl").open("w", encoding="utf-8") as f:
    for k in pilot_keys:
        j = byk[k]
        f.write(json.dumps({"key": k, "attempt": 1, "phase": "pilot", "tag": j["tag"],
                            "messages": j["messages"]}, ensure_ascii=False) + "\n")
# 1-job probe (first pilot key)
j = byk[pilot_keys[0]]
with (HERE / "probe_attempt.jsonl").open("w", encoding="utf-8") as f:
    f.write(json.dumps({"key": j["key"], "attempt": 1, "phase": "probe", "tag": j["tag"],
                        "messages": j["messages"]}, ensure_ascii=False) + "\n")

print("easy jobs_600: 600")
print("pilot keys:")
for k in pilot_keys:
    print("  ", k)
print("probe key:", pilot_keys[0])
