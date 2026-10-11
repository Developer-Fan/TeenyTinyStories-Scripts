"""Build judge job files with the EXACT frozen protocols. Usage:
  python build_jobs.py gpteval1 | gpteval2 | pairwise
eval_outputs/jobs_<phase>.jsonl (shuffled, fixed seeds).
Rules enforced here: no system prompt, no model names/ids, no "TinyStories"
in any message; {PROMPT}***{COMPLETION} with no added space.
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"

G1_HEAD = ("In the following exercise, the student is given a beginning of a story. "
           "The student needs to complete it into a full story. The exercise tests the "
           "student's language abilities and creativity. The symbol *** marks the separator "
           "between the prescribed beginning and the student's completion:\n\n\n")
G1_TAIL = ("\n\n\nPlease provide your general assessment about the part written by the "
           "student (the one after the *** symbol). Is it grammatically correct? Is it "
           "consistent with the beginning of the story? Pay special attention to whether "
           "the student manages to complete the sentence which is split in the middle by "
           "the separator ***.")
G2 = ("Now, grade the student's completion in terms of grammar, creativity, "
      "consistency with the story's beginning and whether the plot makes sense. "
      "Moreover, please provide your best guess of what the age of the student might "
      "be, as reflected from the completion. Choose from possible age groups: A: 3 or "
      "under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.\n\n\n"
      "Reply in exactly this format and nothing else:\n"
      "Grammar: X/10, Creativity: X/10, Consistency: X/10, Age group: X")

PAIR_HEAD = ("In the following exercise, two students are each given the same beginning "
             "of a story. Each student completes it into a full story. The exercise tests "
             "the students' language abilities and creativity. The symbol *** marks the "
             "separator between the prescribed beginning and a student's completion.\n\n\n"
             "Story beginning (the same for both students):\n")
PAIR_MID_A = "\n\n\nStudent A's completion:\n"
PAIR_MID_B = "\n\n\nStudent B's completion:\n"
PAIR_TAIL = ("\n\n\nDecide which student wrote the better completion. Consider: "
             "(1) grammar, including whether the sentence split by *** is completed "
             "correctly; (2) creativity; (3) consistency with the story beginning; "
             "(4) whether the plot makes sense. Do not prefer a completion because of "
             "its length or its position. If the two are of genuinely equal quality, "
             "answer TIE.\n\n\nFirst write one sentence explaining your decision. Then, "
             "on a new final line, write exactly one of:\nVerdict: A\nVerdict: B\n"
             "Verdict: TIE")


from prompts_lib import load_prompts


def load_gens():
    rows = {}
    with open(OUT / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["model"], r["prompt_id"], r["completion_idx"])] = r
    return rows


def check_clean(*texts):
    for t in texts:
        assert "TinyStories" not in t, "leak: TinyStories"
    return True


def phase_gpteval1():
    prompts, gens = load_prompts(), load_gens()
    assert len(gens) == 2000, f"frozen set incomplete: {len(gens)}"
    jobs = []
    for (model, pid, ci), r in gens.items():
        body = f"{G1_HEAD}{prompts[pid]}***{r['text']}{G1_TAIL}"
        check_clean(body)
        jobs.append({"key": f"g1|{model}|{pid}|{ci}",
                     "messages": [{"role": "user", "content": body}],
                     "tag": {"phase": "g1", "model": model, "pid": pid, "ci": ci}})
    random.Random(999).shuffle(jobs)
    with open(OUT / "jobs_gpteval1.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs)} turn-1 jobs")


def phase_gpteval2():
    t1 = {}
    with open(OUT / "judge_raw_gpteval1.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("error"):
                t1[r["key"]] = r["response_text"]
    missing = 0
    jobs = []
    with open(OUT / "jobs_gpteval1.jsonl", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            a1 = t1.get(j["key"])
            if a1 is None:
                missing += 1
                continue
            jobs.append({"key": j["key"].replace("g1|", "g2|", 1),
                         "messages": j["messages"] + [
                             {"role": "assistant", "content": a1},
                             {"role": "user", "content": G2}],
                         "tag": {**j["tag"], "phase": "g2"}})
    random.Random(1000).shuffle(jobs)
    with open(OUT / "jobs_gpteval2.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs)} turn-2 jobs, missing turn-1: {missing}")


def phase_pairwise():
    import itertools
    gens = load_gens()
    assert len(gens) == 2000, f"frozen set incomplete: {len(gens)}"
    prompts = load_prompts()
    models = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
    jobs = []
    for a, b in itertools.combinations(models, 2):
        for pid in range(1, 51):
            ta, tb = gens[(a, pid, 0)]["text"], gens[(b, pid, 0)]["text"]
            for order, (x, y, xname, yname) in (
                    ("ab", (ta, tb, a, b)), ("ba", (tb, ta, b, a))):
                body = (f"{PAIR_HEAD}{prompts[pid]}"
                        f"{PAIR_MID_A}{prompts[pid]}***{x}"
                        f"{PAIR_MID_B}{prompts[pid]}***{y}"
                        f"{PAIR_TAIL}")
                check_clean(body)
                # swap check: identical except A/B slots
                jobs.append({
                    "key": f"p|{a}|{b}|{pid}|{order}",
                    "messages": [{"role": "user", "content": body}],
                    "tag": {"phase": "pair", "a": a, "b": b, "pid": pid,
                            "order": order, "A_is": xname, "B_is": yname}})
    # verify byte-identical-except-slots for every swapped pair
    by = {}
    for j in jobs:
        by.setdefault((j["tag"]["a"], j["tag"]["b"], j["tag"]["pid"]), {})[j["tag"]["order"]] = j["messages"][0]["content"]
    for k, d in by.items():
        a, b, pid = k
        ta, tb = gens[(a, pid, 0)]["text"], gens[(b, pid, 0)]["text"]
        exp_ab = (f"{PAIR_HEAD}{prompts[pid]}"
                  f"{PAIR_MID_A}{prompts[pid]}***{ta}"
                  f"{PAIR_MID_B}{prompts[pid]}***{tb}{PAIR_TAIL}")
        exp_ba = (f"{PAIR_HEAD}{prompts[pid]}"
                  f"{PAIR_MID_A}{prompts[pid]}***{tb}"
                  f"{PAIR_MID_B}{prompts[pid]}***{ta}{PAIR_TAIL}")
        assert d["ab"] == exp_ab and d["ba"] == exp_ba, f"swap mismatch {k}"
    random.Random(1001).shuffle(jobs)
    with open(OUT / "jobs_pairwise.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs)} pairwise jobs (swap-verified)")


if __name__ == "__main__":
    {"gpteval1": phase_gpteval1, "gpteval2": phase_gpteval2,
     "pairwise": phase_pairwise}[sys.argv[1]]()
