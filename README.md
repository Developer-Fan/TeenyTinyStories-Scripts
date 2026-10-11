# Reproducibility scripts (copies)

Copies of every script used, by pipeline stage. Originals live at the paths
noted; behaviour is unchanged except `07_easy_runs` outputs default to `./`
(run with an explicit out path). Judge runners need API keys via environment
(`PUTER_AUTH_TOKEN`, `MISTRAL_KEY1..N`) — keys are never written to disk.

- `01_generation/` — TTS corpus generation (`vast/generate.py`, setup and pod
  run scripts). Filters: unfinished (400-token cap, no EOS) → short (<30 tok)
  → no terminal punctuation → in-shard dedupe; header truncation.
- `02_merge_stats/` — merge (`merge.py`), corpus/token/vocab stats, MinHash
  near-duplicates, notebook builder, checklist helper.
- `03_training/` — data prep (`prepare_data.py`: GPT-Neo tokenize, 512-block
  packing, manifests) and trainers. Canonical run used `train.py`;
  `train_package.py` (`cloud_training/training/train.py`) and
  `train_models_dir.py` (`models/train.py`) are earlier variants, renamed to
  avoid collisions.
- `04_eval/` — eval harness: prompt library, frozen generations, job builders
  (pairwise + two-turn GPT-eval), human packets/raters, aggregators, Mistral
  batch runner (`judge_mistral.py`: `MISTRAL_KEY*` round-robin, N workers).
- `05_judging/` — second-judge (Luna) pipeline: job builders, Puter runner
  (`judge2_run.mjs`: `PUTER_AUTH_TOKEN`, one attempt per line, resume-safe),
  strict parsers, BT/Elo analysis + verifier, human-Elo runners, easy-set
  builders/analysers (`*_easy*`, `*_raw*`, `*egpteval*`, `*lenient*`).
- `06_paper/` — `latex.tex`, `references.bib`, converter. Build needs pdflatex.
- `07_easy_runs/` — easy-prompt generation scripts (rewritten after scratch
  cleanup; logic identical): `easy_gen.py` (2 prompts x10, T=1.0, truncated),
  `easy_gen10.py` (prompts 201-210 x1), `easy2000_gen.py` (50 prompts x10,
  T=0.8, truncated), `easy2000_raw_gen.py` (same, no truncation + diagnostic).
  Rescued raw archive: `analyse/easy_judge_raw/generations_raw.jsonl`
  (rebuilt from job messages, stats verified identical).

Data, checkpoints, judge logs and model outputs are NOT included (see
`results/`, `analyse/`, `merged/`, HuggingFace `ableian/TeenyTinyStories-1M`
and `-33M`). `SHA256SUMS` pins these copies.
