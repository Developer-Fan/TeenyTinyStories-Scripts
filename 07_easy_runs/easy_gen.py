"""Easy-prompt generations, T=1.0 with rep4 truncation (first easy probe).

2 prompts (101: 'Once upon a time,', 102: 'One day, a little') x 10 completions
x 2 models (ours_33M vs ts_33M). Same sampling as eval/generate.py (top_p=1.0,
top_k off, max_new=300, EOS stop, seed pid*1000+ci) plus truncate_rep4.
Output: easy_gens.jsonl (40 rows). Run: python easy_gen.py [out_path]
"""
import json, re, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

WORD_RE = re.compile(r"[a-z]+")
EOS = 50256
TEMP = 1.0
PROMPTS = {101: "Once upon a time,", 102: "One day, a little"}
MODELS = {
    "ours_33M": "cloud_training/results/runs/33m_tts/final",
    "ts_33M": "roneneldan/TinyStories-33M",
}
N = 10
MAX_NEW = 300
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("easy_gens.jsonl")

def truncate_rep4(text):
    spans = [(m.group(0), m.start()) for m in WORD_RE.finditer(text.lower())]
    seen = {}
    for i in range(len(spans) - 3):
        g = (spans[i][0], spans[i+1][0], spans[i+2][0], spans[i+3][0])
        if g in seen:
            return text[:spans[i][1]].rstrip(), True
        seen[g] = i
    return text, False

def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device={dev} temp={TEMP}")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M",
                                        clean_up_tokenization_spaces=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if OUT.exists():
        for line in open(OUT):
            try:
                r = json.loads(line); have.add((r["model"], r["prompt_id"], r["completion_idx"]))
            except Exception: pass
        log(f"resuming: {len(have)} present")
    fout = open(OUT, "a", encoding="utf-8")
    for name, src in MODELS.items():
        model = AutoModelForCausalLM.from_pretrained(src).to(dev).eval()
        for pid, prompt in sorted(PROMPTS.items()):
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            for ci in range(N):
                if (name, pid, ci) in have: continue
                torch.manual_seed(pid * 1000 + ci)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=MAX_NEW,
                                         pad_token_id=EOS, do_sample=True,
                                         temperature=TEMP, top_k=0, top_p=1.0)
                text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                text, trunc = truncate_rep4(text)
                fout.write(json.dumps({"model": name, "prompt_id": pid,
                    "prompt": prompt, "completion_idx": ci, "text": text,
                    "n_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                    "truncated": trunc, "seed": pid*1000+ci, "temp": TEMP}) + "\n")
                fout.flush()
        del model
        if dev.type == "mps": torch.mps.empty_cache()
        log(f"{name} done")
    fout.close(); log("wrote " + str(OUT))

if __name__ == "__main__":
    main()
