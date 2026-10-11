"""Easy-10: first ten easy prompts (ids 201-210), completion idx 0 only, both 33M
models, T=1.0 with rep4 truncation. Seeds pid*1000+ci so x10 can extend later.
Output: easy10_gens.jsonl (20 rows). Run: python easy_gen10.py [out_path]
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
PROMPTS = {
    201: "Once upon a time,",
    202: "One day, a little girl named Lily",
    203: "One day, a little boy named Tim",
    204: "Once there was a tiny kitten",
    205: "One morning, a puppy woke up",
    206: "In a small house lived a bunny",
    207: "One day, a duck found a ball",
    208: "There was a happy frog in the pond",
    209: "A little bird sat in a tree",
    210: "One night, a cat saw the moon",
}
MODELS = {
    "ours_33M": "cloud_training/results/runs/33m_tts/final",
    "ts_33M": "roneneldan/TinyStories-33M",
}
CIS = [0]
MAX_NEW = 300
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("easy10_gens.jsonl")

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
            for ci in CIS:
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
