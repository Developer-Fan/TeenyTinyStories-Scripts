"""Retry loop for GPT-eval parse/transport failures: redo the WHOLE call
(fresh turn-1 + turn-2), up to 3 retries (4 attempts total). Usage:
  python retry_gpteval.py <round>   # round = 1,2,3
Reads gpteval_results.json failed_keys (+ previous retry outputs), emits
jobs_retry<R>_g1.jsonl; after judging, jobs_retry<R>_g2.jsonl is built from
whatever turn-1 responses arrive. aggregate merges by base key, first success wins.
"""
import glob
import json
import re
import sys
from pathlib import Path

RX_STRICT = re.compile(r"^\s*Grammar:\s*(\d{1,2})\s*/\s*10\s*,\s*Creativity:\s*(\d{1,2})"
                        r"\s*/\s*10\s*,\s*Consistency:\s*(\d{1,2})\s*/\s*10\s*,"
                        r"\s*Age group:\s*([A-F])\b")

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"


def true_failing():
    """Reconcile every raw file: keys with no successful strict parse."""
    ok = set()
    last_err = {}
    for p in sorted([OUT / "judge_raw_gpteval2.jsonl"] +
                    [Path(q) for q in glob.glob(str(OUT / "judge_raw_retry*_g2.jsonl"))]):
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            b = r["key"].split("#")[0]
            if b in ok:
                continue
            if not r.get("error") and RX_STRICT.search(
                    (r.get("response_text") or "").replace("**", "")):
                ok.add(b)
                last_err.pop(b, None)
            else:
                last_err[b] = r.get("error") or "parse"
    return ok, last_err


def collect_done():
    """base key -> attempts used so far (across g2 + retry outputs)."""
    from collections import Counter
    c = Counter()
    for p in [OUT / "judge_raw_gpteval2.jsonl"] + sorted(OUT.glob("judge_raw_retry*_g2.jsonl")):
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            base = r["key"].split("#")[0].replace("g1|", "g2|")
            c[base] += 1
    return c


def main():
    rnd = int(sys.argv[1])
    assert 1 <= rnd <= 3
    _, failing = true_failing()
    have_rounds = len(glob.glob(str(OUT / "jobs_retry*_g1.jsonl")))
    todo = sorted(failing)
    print(f"round {rnd}: {len(todo)} still failing; emitting whole-call retries")
    # need original turn-1 messages
    j1 = {}
    with open(OUT / "jobs_gpteval1.jsonl", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            j1[j["key"]] = j
    jobs = []
    for base in todo:
        g1key = base.replace("g2|", "g1|")
        j = j1[g1key]
        jobs.append({"key": f"{g1key}#r{rnd}", "messages": j["messages"],
                     "tag": {**j["tag"], "retry": rnd}})
    with open(OUT / f"jobs_retry{rnd}_g1.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"round {rnd}: {len(jobs)} whole-call retries")


if __name__ == "__main__":
    main()


def build_g2():
    """Build turn-2 jobs for the latest retry round from its turn-1 responses."""
    import glob
    rnds = sorted({p.split("retry")[1][0] for p in glob.glob(str(OUT / "judge_raw_retry*_g1.jsonl"))},
                  reverse=True)
    if not rnds:
        print("no retry g1 outputs")
        return
    rnd = rnds[0]
    from build_jobs import G2
    t1 = {}
    with open(OUT / f"judge_raw_retry{rnd}_g1.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("error"):
                t1[r["key"]] = r["response_text"]
    jobs = []
    with open(OUT / f"jobs_retry{rnd}_g1.jsonl", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j["key"] in t1:
                jobs.append({"key": j["key"].replace("g1|", "g2|", 1),
                             "messages": j["messages"] + [
                                 {"role": "assistant", "content": t1[j["key"]]},
                                 {"role": "user", "content": G2}],
                             "tag": {**j["tag"], "phase": "g2"}})
    import random
    random.Random(1000 + int(rnd)).shuffle(jobs)
    with open(OUT / f"jobs_retry{rnd}_g2.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"round {rnd}: {len(jobs)} turn-2 retry jobs")


if len(sys.argv) > 2 and sys.argv[2] == "g2":
    build_g2()
elif __name__ == "__main__" and len(sys.argv) == 2:
    main()
