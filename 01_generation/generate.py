# ===== TeenyTinyStories production generator =====

import os, re, glob, json, random, time, hashlib
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple Silicon: fall back to CPU for unsupported ops
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

import torch.nn.functional as F
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention

def _sdpa_attn(self, query, key, value, attention_mask=None):
    q_len, k_len = query.size(-2), key.size(-2)
    mask = attention_mask
    if self.attention_type == "local":
        i = torch.arange(k_len - q_len, k_len, device=query.device).unsqueeze(1)
        j = torch.arange(k_len, device=query.device).unsqueeze(0)
        win = (j <= i) & (j > i - self.config.window_size)
        if mask is None:
            mask = win
        elif mask.dtype == torch.bool:
            mask = mask & win
        else:
            mask = mask.masked_fill(~win, torch.finfo(mask.dtype).min)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0,
        is_causal=(mask is None), scale=1.0), None

GPTNeoSelfAttention._attn = _sdpa_attn

MODEL_ID        = "roneneldan/TinyStories-Instruct-33M"
TOKENIZER_ID    = "EleutherAI/gpt-neo-125M"
MODE            = os.environ.get("MODE", "generate")      # generate | merge
SHARD_ID        = int(os.environ.get("SHARD_ID", "0"))   # one shard per process
SEED            = 0              # same everywhere; the shard ID makes the seed streams differ
TARGET_TOKENS   = int(os.environ.get("TARGET_TOKENS", "50000000"))   # per shard; 0 disables the cap (use 0 for merge)
TEMPERATURE     = 1.0
TOP_P           = 1.0            # ablation: ~96-97% of real distinct-n vs ~85-92% at 0.95
BATCH_SIZE      = 256             # 4 parallel processes on one 24 GB RTX 4090
MAX_NEW_TOKENS  = 400
MIN_STORY_TOKENS = 30
FILTER_ON_SEEDS = False          # keep misses; they're data about the student's seed following
OUT_DIR         = "Corpus_TeenyTinyStories"
USE_DRIVE       = False          # Colab-only; leave False on this machine
MAX_MINUTES     = 300             # stop the cell after N minutes; rerun the cell to resume

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, "| mode:", MODE, "| out:", OUT_DIR)

# ---------------- REAL INSTRUCT DATA (seed pools + baselines) ----------------
FIELD = re.compile(r"^(Words|Features|Summary|Random sentence|Story):[ \t]*(.*)$")
WORD  = re.compile(r"[a-z']+")
HDR   = re.compile(r"\n?[ \t]*(Summary|Features|Words|Random sentence|Sentence|Story):")
TERMINAL = re.compile(r'[.!?]["\'”’)]*\s*$')

def parse_instruct(path):
    recs = []
    for block in open(path, encoding="utf-8").read().split("<|endoftext|>"):
        rec, cur = {}, None
        for line in block.strip("\n").split("\n"):
            m = FIELD.match(line)
            if m: cur = m.group(1); rec[cur] = [m.group(2)]
            elif cur is not None: rec[cur].append(line)
        if "Story" in rec:
            recs.append({k: "\n".join(v).strip() for k, v in rec.items()})
    return recs

records = parse_instruct(hf_hub_download("roneneldan/TinyStoriesInstruct",
                         "TinyStories-Instruct-valid.txt", repo_type="dataset"))
word_recs = []
for r in records:
    if r.get("Words"):
        w = [x.strip() for x in r["Words"].split(",") if x.strip()]
        if len(w) == 3: word_recs.append((w, r["Story"]))
assert len(word_recs) > 1000, "no usable 'Words:' lines parsed; check the dataset format"
POS_LISTS = [[w[i] for w, _ in word_recs] for i in range(3)]          # position-wise: verb, noun, adjective
FEATURE_POOL = [[x.strip() for x in r.get("Features", "").split(",") if x.strip()]
                for r in records if r.get("Words")]                    # includes empty sets (~46% of records)
print(f"records {len(records)} | word-seeded {len(word_recs)} | "
      f"feature sets {len(FEATURE_POOL)} (empty: {sum(1 for f in FEATURE_POOL if not f)})")

def sample_seed(rng):
    return {"words": [rng.choice(POS_LISTS[i]) for i in range(3)], "features": list(rng.choice(FEATURE_POOL))}

def make_prompt(s):   # format chosen on a 128-sample probe: 'Words' line, optional 'Features' line, 'Story:'
    f = f"Features: {', '.join(s['features'])}\n" if s["features"] else ""
    return f"Words: {', '.join(s['words'])}\n{f}Story:"

def has_word(text, w):   # lenient: 'dive' ~ 'diving'; slightly overstates adherence (same matcher used for the teacher)
    stem = w[:-1] if (w.endswith("e") and len(w) > 3) else w
    return re.search(r"\b" + re.escape(stem), text.lower()) is not None

def distinct_n(texts, n):
    total, uniq = 0, set()
    for t in texts:
        w = WORD.findall(t.lower()); g = list(zip(*[w[i:] for i in range(n)]))
        total += len(g); uniq.update(g)
    return len(uniq) / max(total, 1)

tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, clean_up_tokenization_spaces=False)
tok.pad_token, tok.padding_side = tok.eos_token, "left"
EOS = tok.eos_token_id

# teacher-side baselines (real GPT-written, word-seeded records), computed once
_rl = sorted(len(x) for x in tok([s for _, s in word_recs])["input_ids"])
TEACHER = {
    "seed_rate": [sum(has_word(s, w[i]) for w, s in word_recs) / len(word_recs) for i in range(3)],
    "seed_all3": sum(all(has_word(s, x) for x in w) for w, s in word_recs) / len(word_recs),
    "len_p5_p50_p95": [_rl[int(q * (len(_rl) - 1))] for q in (.05, .5, .95)],
    "frac_over_cap": sum(l > MAX_NEW_TOKENS for l in _rl) / len(_rl),
}

# ---------------- REPORT ----------------
def report(stories, label, extra=None):
    n = len(stories); texts = [r["story"] for r in stories]
    lens = sorted(r["n_tokens"] for r in stories)
    seeds_all = [r["seed"]["words"] for r in stories]
    hw = [r.get("has_words") or [has_word(r["story"], w) for w in r["seed"]["words"]] for r in stories]
    shuf = seeds_all[:]; random.Random(0).shuffle(shuf)
    chance = [sum(has_word(r["story"], s[i]) for r, s in zip(stories, shuf)) / n for i in range(3)]
    # distinct-n on <=800k-word random subsample vs a real slice of equal size (memory-safe at any corpus size)
    idx = list(range(n)); random.Random(0).shuffle(idx); sub, w = [], 0
    for i in idx:
        sub.append(texts[i]); w += len(WORD.findall(texts[i].lower()))
        if w >= 800_000: break
    nw = sum(len(WORD.findall(t.lower())) for t in sub); real, acc = [], 0
    for rec in records:
        real.append(rec["Story"]); acc += len(WORD.findall(rec["Story"].lower()))
        if acc >= nw: break
    d_gen, d_real = [distinct_n(sub, k) for k in (1, 2, 3)], [distinct_n(real, k) for k in (1, 2, 3)]
    first12 = Counter(" ".join(WORD.findall(t.lower())[:12]) for t in texts)
    rep = {
        "label": label, "stories": n, "story_tokens": sum(lens), "mean_tokens": sum(lens) / n,
        "len_p5_p50_p95": [lens[int(q * (n - 1))] for q in (.05, .5, .95)],
        "header_leakage": sum(bool(HDR.search(t)) for t in texts),
        "terminal_punct_rate": sum(bool(TERMINAL.search(t)) for t in texts) / n,
        "student_seed_all3": sum(all(h) for h in hw) / n,
        "student_seed_rate": [sum(h[i] for h in hw) / n for i in range(3)],
        "chance_seed_rate_shuffled": chance,
        "teacher": TEACHER,
        "distinct_1_2_3_generated": d_gen, "distinct_1_2_3_real": d_real,
        "distinct_ratio_vs_real": [a / b for a, b in zip(d_gen, d_real)],
        "stories_sharing_first12_words": sum(c for c in first12.values() if c > 1),
    }
    if extra: rep.update(extra)
    json.dump(rep, open(f"{OUT_DIR}/summary_{label}.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    for r in random.Random(1).sample(stories, 3):
        print("\n--- seed:", r["seed"], "\n", r["story"][:700])

# ---------------- MODE: GENERATE ----------------
if MODE == "generate":
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()   # fp32 on purpose

    @torch.no_grad()
    def generate_batch(seeds):
        enc = tok([make_prompt(s) for s in seeds], return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=0,
                             max_new_tokens=MAX_NEW_TOKENS, pad_token_id=EOS, eos_token_id=EOS)
        res = []
        for s, row in zip(seeds, out[:, enc["input_ids"].shape[1]:].cpu().tolist()):
            finished = EOS in row
            if finished: row = row[:row.index(EOS)]
            text = tok.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            m = HDR.search(text)
            if m: text, finished = text[:m.start()], True     # next record's header began: cut it off
            text = text.strip()
            res.append({"seed": s, "story": text, "finished": finished,
                        "n_tokens": len(tok(text)["input_ids"]) if text else 0})
        return res

    cfg = dict(MODEL_ID=MODEL_ID, TOKENIZER_ID=TOKENIZER_ID, TEMPERATURE=TEMPERATURE, TOP_P=TOP_P,
               BATCH_SIZE=BATCH_SIZE, MAX_NEW_TOKENS=MAX_NEW_TOKENS, MIN_STORY_TOKENS=MIN_STORY_TOKENS,
               SEED=SEED, SHARD_ID=SHARD_ID, FILTER_ON_SEEDS=FILTER_ON_SEEDS, prompt_example=make_prompt(sample_seed(random.Random(0))))
    json.dump(cfg, open(f"{OUT_DIR}/run_config_shard{SHARD_ID:02d}.json", "w"), indent=2)

    shard_path = f"{OUT_DIR}/stories_shard{SHARD_ID:02d}.jsonl"
    stories, seen, total, stats, rej_tails = [], set(), 0, Counter(), []
    if os.path.exists(shard_path):
        for line in open(shard_path):
            r = json.loads(line); stories.append(r); seen.add(r["story"]); total += r["n_tokens"]
        print(f"resumed shard {SHARD_ID}: {len(stories)} stories, {total:,} tokens")
    rng = random.Random(f"{SEED}-{SHARD_ID}-{len(stories)}")          # fresh seed stream after each resume
    t0 = time.time()
    with open(shard_path, "a") as f_out:
        while total < TARGET_TOKENS:
            if MAX_MINUTES and (time.time() - t0) / 60 > MAX_MINUTES:
                print("time limit reached; stopping (rerun the cell to resume)"); break
            seeds = [sample_seed(rng) for _ in range(BATCH_SIZE)]
            torch.manual_seed(rng.randrange(2**31))
            for r in generate_batch(seeds):
                stats["generated"] += 1
                if not r["finished"]:                   stats["rej_unfinished"] += 1; continue
                if r["n_tokens"] < MIN_STORY_TOKENS:    stats["rej_short"] += 1; continue
                if not TERMINAL.search(r["story"]):
                    stats["rej_no_terminal"] += 1
                    if len(rej_tails) < 10: rej_tails.append(r["story"][-80:])
                    continue
                if r["story"] in seen:                  stats["rej_duplicate"] += 1; continue
                r["has_words"] = [has_word(r["story"], w) for w in r["seed"]["words"]]
                if FILTER_ON_SEEDS and not all(r["has_words"]): stats["rej_seed_miss"] += 1; continue
                r["id"], r["shard"] = len(stories), SHARD_ID
                seen.add(r["story"]); stories.append(r); total += r["n_tokens"]
                f_out.write(json.dumps(r) + "\n")
                if total >= TARGET_TOKENS: break
            f_out.flush()
            el = time.time() - t0
            print(f"{total:>11,} / {TARGET_TOKENS:,} tok | {len(stories):,} stories | {el:.0f}s | "
                  f"{(total / el):.0f} tok/s (session avg incl. resumed) | {dict(stats)}")
    report(stories, f"shard{SHARD_ID:02d}", extra={
        "session_rejections": dict(stats),
        "session_unfinished_rate": stats["rej_unfinished"] / max(stats["generated"], 1),
        "rejected_no_terminal_tails": rej_tails})

# ---------------- MODE: MERGE ----------------
elif MODE == "merge":
    files = sorted(glob.glob(f"{OUT_DIR}/stories_shard*.jsonl"))
    print("shards:", files)
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    seen, merged, total, dups = set(), [], 0, 0
    for fp in files:
        for line in open(fp):
            r = json.loads(line); k = hashlib.md5(norm(r["story"]).encode()).hexdigest()
            if k in seen: dups += 1; continue
            seen.add(k); merged.append(r); total += r["n_tokens"]
            if TARGET_TOKENS and total >= TARGET_TOKENS: break
        if TARGET_TOKENS and total >= TARGET_TOKENS: break
    for i, r in enumerate(merged): r["id"] = i
    with open(f"{OUT_DIR}/stories_merged.jsonl", "w") as f:
        for r in merged: f.write(json.dumps(r) + "\n")
    with open(f"{OUT_DIR}/tts_corpus.txt", "w") as f:        # plain training file, TinyStories-style separators
        for r in merged: f.write(r["story"] + "\n<|endoftext|>\n")
    print(f"merged {len(merged):,} stories, {total:,} tokens | cross-shard duplicates dropped: {dups}")
    report(merged, "merged", extra={"cross_shard_duplicates_dropped": dups})