"""Easy-2000 RAW: same design as easy2000_gen.py (50 prompts 201-250 x 4 models x
10 completions, T=0.8, same seeds pid*1000+ci) but NO rep4 truncation: raw
EOS/max-token text, per Eldan & Li (2023) section 3.1 (truncate only baseline
GPT-Neo/GPT-2 generations, not TS-trained models). Records would_truncate
diagnostic (rep4 cut point) without cutting.
Output: easy2000_raw_gens.jsonl (2000 rows). Run: python easy2000_raw_gen.py [out_path]
"""
import json, re, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

WORD_RE = re.compile(r"[a-z]+")
EOS = 50256
TEMP = 0.8
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
    211: "Lucy had a red balloon",
    212: "Sam found a big box",
    213: "Mia lost her shoe in the park",
    214: "Ben saw a fish in the lake",
    215: "Anna baked cookies with her mom",
    216: "Tim played with his toy car",
    217: "Lily sang a song in the garden",
    218: "A bear woke up in the forest",
    219: "A mouse ran under the table",
    220: "The dog barked at the door",
    221: "It was a sunny day at the beach",
    222: "It rained all morning long",
    223: "Snow fell softly on the trees",
    224: "The wind blew the leaves away",
    225: "A rainbow came out after the rain",
    226: "Lily went to school with her friend",
    227: "Tim gave his teacher an apple",
    228: "The children played hide and seek",
    229: "Mom read a book to Lily",
    230: "Dad took Tim to the farm",
    231: "Grandma told a funny story",
    232: "The baby laughed at the puppy",
    233: "A fish swam round and round",
    234: "The cow ate grass in the field",
    235: "A horse ran across the meadow",
    236: "The monkey climbed up high",
    237: "An owl sat on a branch at night",
    238: "The bees buzzed near the flowers",
    239: "Lily planted a seed in the dirt",
    240: "Tim dug a hole in the sand",
    241: "The cake was sweet and warm",
    242: "Lily drank cold milk for breakfast",
    243: "Tim shared his bread with a friend",
    244: "The bell rang and the kids ran out",
    245: "A kite flew high in the sky",
    246: "The boat sailed down the river",
    247: "Tim kicked the ball very far",
    248: "Lily danced to the music",
    249: "The baby slept in the soft bed",
    250: "Everyone clapped and cheered loudly",
}
MODELS = {
    "ours_1M": "cloud_training/results/runs/1m_tts/final",
    "ours_33M": "cloud_training/results/runs/33m_tts/final",
    "ts_1M": "roneneldan/TinyStories-1M",
    "ts_33M": "roneneldan/TinyStories-33M",
}
N = 10
MAX_NEW = 300
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("easy2000_raw_gens.jsonl")

def rep4_cut(text):
    """Return char index where truncate_rep4 would cut, or None."""
    spans = [(m.group(0), m.start()) for m in WORD_RE.finditer(text.lower())]
    seen = {}
    for i in range(len(spans) - 3):
        g = (spans[i][0], spans[i+1][0], spans[i+2][0], spans[i+3][0])
        if g in seen:
            return spans[i][1]
        seen[g] = i
    return None

def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device={dev} temp={TEMP} RAW (no truncation)")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M",
                                        clean_up_tokenization_spaces=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if OUT.exists():
        for line in open(OUT):
            try:
                r = json.loads(line); have.add((r["model"], r["prompt_id"], r["completion_idx"]))
            except Exception: pass
        log(f"resuming: {len(have)}/2000 present")
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
                cut = rep4_cut(text)
                fout.write(json.dumps({"model": name, "prompt_id": pid,
                    "prompt": prompt, "completion_idx": ci, "text": text,
                    "n_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                    "would_truncate": cut is not None, "cut_char": cut,
                    "seed": pid*1000+ci, "temp": TEMP}) + "\n")
                fout.flush()
        del model
        if dev.type == "mps": torch.mps.empty_cache()
        log(f"{name} done")
    fout.close(); log("ALL DONE " + str(OUT))

if __name__ == "__main__":
    main()
