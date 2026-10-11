"""Finalize: Elo figure (CI bars), LaTeX tables, README. Rerunnable; adds human
side-by-side automatically if human_analysis.json exists."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"
MODELS = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
LAB = {"ours_1M": "ours 1M", "ours_33M": "ours 33M", "ts_1M": "TS 1M", "ts_33M": "TS 33M"}

elo = json.load(open(OUT / "elo_results.json"))
gpt = json.load(open(OUT / "gpteval_results.json"))

fig, ax = plt.subplots(figsize=(7, 3.6))
series = [("judge-300", {m: elo["elo"][m] for m in MODELS})]
hp = OUT / "human_analysis.json"
if hp.exists():
    ha = json.load(open(hp))
    series.append(("human-50", ha["human_elo"]))
    # judge-50 recomputed in analyze step; stored there if present
    if "judge50_elo" in ha:
        series.append(("judge-50", {m: {"elo": ha["judge50_elo"][m]} for m in MODELS}))
for i, (name, d) in enumerate(series):
    xs = np.arange(len(MODELS)) + (i - (len(series) - 1) / 2) * 0.22
    ys = [d[m]["elo"] for m in MODELS]
    lo = [d[m].get("lo", d[m]["elo"]) for m in MODELS]
    hi = [d[m].get("hi", d[m]["elo"]) for m in MODELS]
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                fmt="o", ms=4, capsize=3, label=name)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([LAB[m] for m in MODELS])
ax.set_ylabel("Elo-scale (TS 1M = 1000)")
ax.set_title("Pairwise ratings with 95% CIs")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "elo_ratings.png", dpi=150)

# LaTeX: gpteval
t = ("\\begin{table}[t]\n\\centering\n\\caption{Absolute GPT-eval scores "
     "(ministral-14b-latest; mean over 50 prompts of per-prompt means, 95\\% "
     "bootstrap CI). Scores are not comparable with the original paper (different "
     "judge).}\\label{tab:gpteval}\n\\begin{tabular}{lccc}\n\\toprule\n"
     "Model & Grammar & Creativity & Consistency \\\\\n\\midrule\n")
for m in MODELS:
    v = gpt["models"][m]
    t += (f"{LAB[m]} & {v['g']['mean']:.2f} [{v['g']['lo']:.2f}, {v['g']['hi']:.2f}] & "
          f"{v['c']['mean']:.2f} [{v['c']['lo']:.2f}, {v['c']['hi']:.2f}] & "
          f"{v['s']['mean']:.2f} [{v['s']['lo']:.2f}, {v['s']['hi']:.2f}] \\\\\n")
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(OUT / "gpteval_table.tex").write_text(t)

# LaTeX: elo
e = elo["elo"]
t = ("\\begin{table}[t]\n\\centering\n\\caption{Elo-scale pairwise ratings "
     "(Bradley-Terry on 300 order-consolidated verdicts, TS 1M anchored at 1000; "
     "95\\% CI by bootstrap over prompts).}\\label{tab:elo}\n"
     "\\begin{tabular}{lcc}\n\\toprule\nModel & Rating & 95\\% CI \\\\\n\\midrule\n")
for m in MODELS:
    t += f"{LAB[m]} & {e[m]['elo']:.0f} & [{e[m]['lo']:.0f}, {e[m]['hi']:.0f}] \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
w = elo["win_matrix"]
t += "% win matrix (row wins / 50): " + "; ".join(
    f"{k} {v['a_wins']:.1f}-{v['b_wins']:.1f}" for k, v in w.items()) + "\n"
(OUT / "elo_table.tex").write_text(t)

# README
import datetime
readme = f"""# Eval pipeline (GPT-eval + pairwise Elo + human packet)

## What was run
- Models: ours_1M, ours_33M (local finals), ts_1M/ts_33M (roneneldan/TinyStories-1M/-33M).
- Prompts: cloud_training/prompts_official_style.txt (user's frozen 50,
  sha005759a4, SHA-gated in prompts_lib.py; max_new_tokens=min(300, 512-prompt_tokens)).
  NOTE (2026-10-08): a first full run used an agent-written 50-prompt set, then a
  second run used a 50-prompt user-pasted set (prompts_final.txt); both were archived
  (eval_outputs_ARCHIVE_WRONG_PROMPTS_20261008/) with README_INVALID.txt and the
  pipeline was fully re-run from Step 1 on the official-style set. Nothing from the
  archived runs is cited or merged.
- Generations: 4 models x 50 prompts x 10 (T=1.0, top-k off, top-p 1.0,
  max_new_tokens=min(300, 512-prompt_tokens), seed=pid*1000+ci, MPS).
- Judge change (logged): started with infron:openai/gpt-5.6-luna (417 turn-1 cached, then
  QUARANTINED in eval_outputs/quarantine_gpt56/ — unused). All reported scores use
  mistral:ministral-14b-latest, temperature 0 (supported), on {datetime.date.today()} via
  direct Mistral API with 4 rotating keys (per-key call counts in judge logs is not
  tracked; total calls below).
- Parsing: strip `**` markers only (Ministral bolds scores; user-approved), then the exact
  spec regexes; per-item norm flag in aggregation inputs. Verdict = last matching line.
- Retries: whole-call redo up to 3x. GPT-eval: 2000/2000 parsed, 0 dropped (17 needed
  round 2 for deviant schemas). Pairwise: 600/600 (1 whole-call retry).
- T=0 is not fully deterministic on this judge (same item re-graded 3/5/3/B then 4/3/2/C).

## Calls
- GPT-eval: 2000 turn-1 + 2000 turn-2 + 101 + 2 + 17x2 retry calls.
- Pairwise: 600 + 23 redo + 1 retry.
- Absolute scores NOT comparable with the original paper (different judge).

## Key findings (unhedged numbers; hedge in prose)
- Elo (weak L2 prior 0.01 on BT thetas; conversion rating=1000+400/ln10*theta):
  ours_1M {e['ours_1M']['elo']:.0f}, ours_33M {e['ours_33M']['elo']:.0f}, ts_1M 1000, ts_33M {e['ts_33M']['elo']:.0f}.
  Implied win probs match observed rates within 0.03 on all 6 pairs (checked).
- 1M pair is a tie (CIs overlap); 33M pair goes to released (CIs disjoint).
- Cross-family gaps (1M vs 33M) are fragile: near-sweeps leave the link almost
  unidentified, CIs span ~100-200 points. Do NOT quote the 1M-vs-33M gap as a number.
- GPT-eval means (g/c/s): ours_1M {gpt['models']['ours_1M']['g']['mean']:.2f}/{gpt['models']['ours_1M']['c']['mean']:.2f}/{gpt['models']['ours_1M']['s']['mean']:.2f}, ts_33M {gpt['models']['ts_33M']['g']['mean']:.2f}/{gpt['models']['ts_33M']['c']['mean']:.2f}/{gpt['models']['ts_33M']['s']['mean']:.2f} (see table).
- Position bias: second-listed wins 345/598 non-tie (57.7%); order disagreement 49/300.
- Ties: 2/600 judge verdicts. No majority cycle. Length-win corr -0.097.

## Rerun
All judge calls cached by key in judge_raw_*.jsonl; reruns cost nothing.
python build_jobs.py gpteval1|gpteval2|pairwise; judge via judge_mistral.py \
  <jobs> <out> <workers> with MISTRAL_KEY1..4 in env.
"""
(OUT / "README.md").write_text(readme)
print("wrote elo_ratings.png gpteval_table.tex elo_table.tex README.md")
