"""Build Ministral GPT-eval jobs for the easy RAW stories (T=0.8, no truncation).

Turn 1 (assessment): 2000 jobs, keys eg1|model|pid|ci, same G1_HEAD/G1_TAIL as
cloud_training/eval/build_jobs.py, {PROMPT}***{COMPLETION} no added space.
Turn 2 built after turn 1 completes (needs assistant replies).
Writes analyse/easy_judge_raw/jobs_egpteval1.jsonl
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = Path("/Users/yanfu/Documents/TeenyTinyStories/cloud_training/eval")
RAW_GENS = Path("/var/folders/z8/4b7hbhhn259d_hvt7zl7cw340000gp/T/opencode/easy2000_raw_gens.jsonl")
sys.path.insert(0, str(EVAL))
import build_jobs as B  # noqa: E402 G1_HEAD/G1_TAIL/G2 constants

G2 = B.G2

rows = {}
for line in RAW_GENS.open(encoding="utf-8"):
    r = json.loads(line)
    rows[(r["model"], int(r["prompt_id"]), int(r["completion_idx"]))] = r
assert len(rows) == 2000, len(rows)

jobs = []
for (model, pid, ci), r in rows.items():
    body = f"{B.G1_HEAD}{r['prompt']}***{r['text']}{B.G1_TAIL}"
    assert "TinyStories" not in body
    jobs.append({"key": f"eg1|{model}|{pid}|{ci}",
                 "messages": [{"role": "user", "content": body}],
                 "tag": {"phase": "eg1", "model": model, "pid": pid, "ci": ci}})
random.Random(999).shuffle(jobs)
with (HERE / "jobs_egpteval1.jsonl").open("w", encoding="utf-8") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(f"wrote {len(jobs)} turn-1 jobs")


def build_g2(raw1_path, out_name):
    t1 = {}
    for line in open(raw1_path, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error"):
            t1[r["key"]] = r["response_text"]
    missing = 0
    jobs2 = []
    for line in open(HERE / "jobs_egpteval1.jsonl", encoding="utf-8"):
        j = json.loads(line)
        a1 = t1.get(j["key"])
        if a1 is None:
            missing += 1
            continue
        jobs2.append({"key": j["key"].replace("eg1|", "eg2|", 1),
                      "messages": j["messages"] + [
                          {"role": "assistant", "content": a1},
                          {"role": "user", "content": G2}],
                      "tag": {**j["tag"], "phase": "eg2"}})
    random.Random(1000).shuffle(jobs2)
    with open(HERE / out_name, "w", encoding="utf-8") as f:
        for j in jobs2:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs2)} turn-2 jobs, missing turn-1: {missing}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "g2":
        build_g2(HERE / "judge_raw_egpteval1.jsonl", "jobs_egpteval2.jsonl")
