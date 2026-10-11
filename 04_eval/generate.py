"""STEP 1: frozen generations. 4 models x 50 prompts x 10 completions.

T=1.0, top-k off, top-p 1.0, max_new_tokens=300, stop at EOS (50256).
Seed(prompt_id, completion_idx) = prompt_id * 1000 + completion_idx.
Truncation: cut BEFORE the second occurrence of the first repeated word
4-gram (lowercase [a-z]+ words), as in the original paper's GPT-Neo handling.
Writes eval_outputs/generations.jsonl + .sha256. MPS on Mac, else CPU.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"
OUT.mkdir(exist_ok=True)
T0 = time.time()
WORD_RE = re.compile(r"[a-z]+")
EOS = 50256

MODELS = {
    "ours_1M": "/Users/yanfu/Documents/TeenyTinyStories/cloud_training/results/runs/1m_tts/final",
    "ours_33M": "/Users/yanfu/Documents/TeenyTinyStories/cloud_training/results/runs/33m_tts/final",
    "ts_1M": "roneneldan/TinyStories-1M",
    "ts_33M": "roneneldan/TinyStories-33M",
}
N_COMPLETIONS = 10
MAX_NEW = 300


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


from prompts_lib import load_prompts


def truncate_rep4(text):
    """Cut text at the start of the 2nd occurrence of the first repeated
    word 4-gram (words = lowercase [a-z]+). Returns (text, was_cut)."""
    spans = [(m.group(0), m.start()) for m in WORD_RE.finditer(text.lower())]
    seen = {}
    for i in range(len(spans) - 3):
        g = (spans[i][0], spans[i + 1][0], spans[i + 2][0], spans[i + 3][0])
        if g in seen:
            return text[: spans[i][1]].rstrip(), True
        seen[g] = i
    return text, False


def main():
    prompts = load_prompts()
    import hashlib
    from prompts_lib import PATH, EXPECTED
    log(f"prompts_final.txt sha256: {hashlib.sha256(PATH.read_bytes()).hexdigest()} "
        f"(expected {EXPECTED})")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device={dev}")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M",
                                        clean_up_tokenization_spaces=False)
    # context-limit rule: max_new_tokens = min(300, 512 - prompt_tokens)
    prompt_lens = {pid: len(tok(t, add_special_tokens=False)["input_ids"])
                   for pid, t in prompts.items()}
    import statistics
    log(f"prompt tokens min/mean/max: {min(prompt_lens.values())}/"
        f"{statistics.mean(prompt_lens.values()):.1f}/{max(prompt_lens.values())}")
    under = {pid: 512 - n for pid, n in prompt_lens.items() if 512 - n < 100}
    if under:
        log(f"WARNING prompts leaving <100 new tokens: {under}")
    else:
        log("all prompts leave >=100 new tokens; max_new_tokens=300 for all")
    n_done = 0
    have = set()
    if (OUT / "generations.jsonl").exists():
        import json as _json
        with open(OUT / "generations.jsonl") as f:
            for line in f:
                try:
                    r = _json.loads(line)
                    have.add((r["model"], r["prompt_id"], r["completion_idx"]))
                except Exception:
                    pass
        n_done = len(have)
        log(f"resuming: {n_done} rows already present (skipped, not duplicated)")
    fout = open(OUT / "generations.jsonl", "a", encoding="utf-8")
    for name, src in MODELS.items():
        model = AutoModelForCausalLM.from_pretrained(src).to(dev).eval()
        for pid, prompt in sorted(prompts.items()):
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            for ci in range(N_COMPLETIONS):
                if (name, pid, ci) in have:
                    continue
                seed = pid * 1000 + ci
                # MPS has no per-call Generator support for generate sampling:
                # seed the global RNG (affects MPS) for determinism.
                torch.manual_seed(seed)
                max_new = min(MAX_NEW, 512 - prompt_lens[pid])
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=max_new,
                                         pad_token_id=EOS, do_sample=True,
                                         temperature=1.0, top_k=0, top_p=1.0)
                text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                text, trunc = truncate_rep4(text)
                fout.write(json.dumps({"model": name, "prompt_id": pid,
                                       "completion_idx": ci, "text": text,
                                       "max_new_tokens": max_new,
                                       "n_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                                       "truncated": trunc, "seed": seed}) + "\n")
                fout.flush()
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
        log(f"{name} done")
    fout.close()
    h = hashlib.sha256(open(OUT / "generations.jsonl", "rb").read()).hexdigest()
    (OUT / "generations.jsonl.sha256").write_text(h + "\n")
    log("generations sha256: " + h)


if __name__ == "__main__":
    main()
