"""Build train_1m_tts.ipynb (Part H). Static validation only; never executes cells."""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "train_1m_tts.ipynb"


def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": src.splitlines(keepends=True)}


MD0 = """# Train the 8 x 64 GPT-Neo (TinyStories-1M architecture) on TeenyTinyStories

**NOT YET EXECUTED.**

Purpose: first test run of the matched student programme. Train the 8-layer x 64-wide GPT-Neo from random initialization on the TTS training split (`CORPUS="tts"`, `MATCH_MODE="full_epoch"`, one epoch), evaluating held-out **real TinyStories** validation loss. The matched TS arm reuses this notebook with `CFG["CORPUS"]="ts"`.

Run order: cells 0-7, read the PLAN output, then optional 8 (benchmark) and 9 (LR sweep), then 10 (training), 11 (final), 12 (checklist). No cell starts training silently; cell 10 is the only one that trains.

Heavy cells: **5** (tokenization of ~500M tokens), **10** (training), **8** (benchmark, only if enabled). Cell 3 hashes ~4.5 GB (moderate). Everything else is cheap.
"""

C1 = """import json, platform, shutil, sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "models" else Path.cwd()
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

try:
    import torch, transformers, tokenizers, datasets, numpy, huggingface_hub
    VERSIONS = {"torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "datasets": datasets.__version__,
                "numpy": numpy.__version__, "huggingface_hub": huggingface_hub.__version__}
except Exception as exc:
    print("Missing dependency:", exc)
    print("Install example (pin to the versions recorded in TRAINING_CHECKLIST.md):")
    print("  pip install torch transformers tokenizers datasets numpy huggingface_hub")
    raise

GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
CUDA = torch.version.cuda if torch.cuda.is_available() else None
FREE = shutil.disk_usage(PROJECT_ROOT).free
NEED = 6 * 1024 ** 3
assert FREE > NEED, f"need > {NEED / 1e9:.1f} GB free, have {FREE / 1e9:.1f} GB"

env = {"python": sys.version.split()[0], "platform": platform.platform(), "gpu": GPU, "cuda": CUDA,
       "mps": torch.backends.mps.is_available(),
       "free_disk_gb": round(FREE / 1e9, 2), "packages": VERSIONS, "cwd": str(Path.cwd()),
       "project_root": str(PROJECT_ROOT)}
(RESULTS / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
print(json.dumps(env, indent=2))
assert torch.cuda.is_available() or torch.backends.mps.is_available(), "no CUDA or MPS accelerator found"
"""

C2 = '''CFG = {
    "RUN_ID": "8x64_tts_seed0",                   # DECIDED: first test is 8x64 on TTS
    "CORPUS": "tts",                              # DECIDED: "tts" (first test) or "ts" (matched arm)

    "PROJECT_ROOT": str(PROJECT_ROOT),            # MEASURED
    "TTS_JSONL": "merged/stories_merged.jsonl",   # MEASURED: 2,511,490 stories / 500,000,803 tokens
    "TTS_TXT": "merged/tts_corpus.txt",           # MEASURED: same stories, EOS separated
    "TS_DATASET": "roneneldan/TinyStories",       # CONFIG: revision f54c09fd..., license cdla-sharing-1.0
    "RESULTS_DIR": "results",                     # DECIDED
    "DATA_DIR": "results/data",                   # DECIDED: tokenized memmaps
    "CKPT_DIR": "results/checkpoints",            # DECIDED

    "HUB_REPO": "roneneldan/TinyStories-1M",      # CONFIG: 8 x 64, 16 heads, window 256, vocab 50257
    "SEQ_LEN": 512,                               # PROPOSED
    "MAX_STORY_TOKENS": 400,                      # DECIDED: drop longer stories (TTS cap confound)

    "DATA_SEED": 1234,                            # DECIDED: story-level shuffle and 20k TTS val split
    "VAL_STORIES": 20000,                         # DECIDED
    "PACKING": "concat_with_eos",                 # PROPOSED: no cross-story attention masking (explicit choice)
    "MATCH_MODE": "full_epoch",                   # DECIDED default; "equal_steps" recommended alternative
    "TTS_TRAIN_TOKENS_WITH_EOS": 498492043,       # MEASURED: reference for equal_steps

    "TOKENS_PER_STEP": 65536,                     # PROPOSED: options 32768 / 65536 / 131072
    "MICRO_BATCH": 8,                             # PENDING: choose from benchmark cell 8
    "LR": 1e-3,                                   # PROPOSED
    "BETAS": (0.9, 0.95),                         # PROPOSED
    "EPS": 1e-8,                                  # PROPOSED
    "WEIGHT_DECAY": 0.1,                          # PROPOSED: matrices only (no biases/LayerNorm/embeddings)
    "GRAD_CLIP": 1.0,                             # PROPOSED
    "WARMUP_FRAC": 0.01,                          # PROPOSED
    "MIN_LR_FRAC": 0.1,                           # PROPOSED
    "DEVICE": "auto",                             # DECIDED: auto -> cuda, then mps, then cpu (runs on MPS)
    "AUTOCAST": "auto",                           # PROPOSED: auto = bf16 on CUDA, off (fp32) on MPS; "fp16" opts in on MPS
    "COMPILE": False,                             # PROPOSED: available in the benchmark cell (CUDA only)

    "INIT_SEED": 0,                               # PROPOSED
    "EVAL_SEED": 7,                               # PROPOSED
    "EVAL_EVERY_STEPS": 250,                      # PROPOSED
    "EVAL_SUBSET_TOKENS": 262144,                 # PROPOSED: fixed subset during training
    "CKPT_EVERY_STEPS": 500,                      # PROPOSED
    "KEEP_LAST_K": 2,                             # PROPOSED
    "RESUME_PATH": None,                          # PROPOSED: set to a checkpoint dir to resume exactly

    "RUN_BENCHMARK": False,                       # DECIDED: benchmark cell is opt-in
    "RUN_LR_SWEEP": False,                        # DECIDED: sweep cell is opt-in and TS-only
    "USE_DRIVE": False,                           # optional Colab Drive mount
    "DRIVE_DIR": "/content/drive/MyDrive/TeenyTinyStories",
}
if CFG["USE_DRIVE"]:
    from google.colab import drive
    drive.mount("/content/drive")
    CFG["DATA_DIR"] = CFG["DRIVE_DIR"] + "/data"
    CFG["CKPT_DIR"] = CFG["DRIVE_DIR"] + "/checkpoints"

GRAD_ACCUM = CFG["TOKENS_PER_STEP"] // (CFG["MICRO_BATCH"] * CFG["SEQ_LEN"])
assert GRAD_ACCUM >= 1, "MICRO_BATCH * SEQ_LEN must not exceed TOKENS_PER_STEP"
print(json.dumps({k: v for k, v in CFG.items() if k not in ("BETAS",)}, indent=2, default=str))
print("GRAD_ACCUM =", GRAD_ACCUM)
'''

C3 = '''import hashlib

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()

stats = json.loads((PROJECT_ROOT / "results/corpus_stats.json").read_text(encoding="utf-8"))
base = json.loads((PROJECT_ROOT / "results/baseline_ts.json").read_text(encoding="utf-8"))

if CFG["CORPUS"] == "tts":
    for key in ("merged_jsonl", "merged_txt"):
        info = stats["inputs"][key]
        p = Path(info["path"])
        assert p.exists(), p
        actual = sha256(p)
        assert actual == info["sha256"], f"hash mismatch for {p}"
        print(f"{p.name}: {p.stat().st_size:,} bytes | sha256 OK")
    n_jsonl = sum(1 for _ in open(PROJECT_ROOT / CFG["TTS_JSONL"], "rb"))
    n_txt = (PROJECT_ROOT / CFG["TTS_TXT"]).read_text(encoding="utf-8").count("<|endoftext|>")
    assert n_jsonl == stats["merged"]["stories"] == 2511490, n_jsonl
    assert n_txt == n_jsonl, (n_txt, n_jsonl)
    print(f"stories {n_jsonl:,} | tokens (no EOS) {stats['merged']['tokens_no_eos']:,}")
else:
    from huggingface_hub import hf_hub_download
    for split, key in (("train", "train"), ("valid", "valid")):
        fname = "TinyStories-train.txt" if split == "train" else "TinyStories-valid.txt"
        p = Path(hf_hub_download(CFG["TS_DATASET"], fname, repo_type="dataset"))
        actual = sha256(p)
        assert actual == base[key]["sha256"], f"hash mismatch for {p}"
        print(f"{p.name}: {p.stat().st_size:,} bytes | sha256 OK")
    print(f"TS train {base['train']['stories']:,} stories | {base['train']['tokens_with_eos']:,} tokens (EOS)")
'''

C4 = '''from transformers import GPTNeoConfig, GPTNeoForCausalLM
from huggingface_hub import hf_hub_download
import torch

raw_path = hf_hub_download(CFG["HUB_REPO"], "config.json")
raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
(Path(CFG["RESULTS_DIR"]) / "model_config_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

config = GPTNeoConfig(**{k: v for k, v in raw.items() if not k.startswith("_") and k != "architectures"})
assert config.num_layers == 8, config.num_layers
assert config.hidden_size == 64, config.hidden_size
assert config.num_heads == 16, config.num_heads
assert config.window_size == 256, config.window_size
assert config.vocab_size == 50257, config.vocab_size
assert config.attention_types == [[["global", "local"], 4]], config.attention_types

torch.manual_seed(CFG["INIT_SEED"])
model = GPTNeoForCausalLM(config)
total = sum(p.numel() for p in model.parameters())
emb = config.vocab_size * config.hidden_size
info = {"hub_repo": CFG["HUB_REPO"], "config_raw": raw, "params_total": total,
        "params_embeddings": emb, "params_non_embedding": total - emb, "init_seed": CFG["INIT_SEED"]}
(Path(CFG["RESULTS_DIR"]) / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"params total={total:,} | embeddings={emb:,} | non-embedding={total - emb:,}")
'''

C5 = '''import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", clean_up_tokenization_spaces=False)
EOS = tok.eos_token_id
assert EOS == 50256
VOCAB = CFG["vocab_size"] if "vocab_size" in CFG else 50257
SEQ = CFG["SEQ_LEN"]
DATA_DIR = Path(CFG["DATA_DIR"])
DATA_DIR.mkdir(parents=True, exist_ok=True)


def pack_stories(ids_iter, out_path):
    tmp = Path(str(out_path) + ".tmp")
    buf, written, stories, counted = [], 0, 0, 0
    with open(tmp, "wb") as f:
        for ids in ids_iter:
            buf.extend(ids)
            buf.append(EOS)
            counted += len(ids) + 1
            stories += 1
            while len(buf) >= SEQ:
                np.asarray(buf[:SEQ], dtype=np.uint16).tofile(f)
                del buf[:SEQ]
                written += SEQ
    tmp.rename(str(out_path))
    return {"path": str(out_path), "stories": stories, "counted_tokens": counted,
            "written_tokens": written, "dropped_tail": counted - written}


def tokenize_story(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert all(i < VOCAB for i in ids), "token id outside vocab"
    return ids


def tts_stories():
    text = (PROJECT_ROOT / CFG["TTS_TXT"]).read_text(encoding="utf-8")
    return [b.strip() for b in text.split("<|endoftext|>") if b.strip()]


def ts_stories():
    path = Path(hf_hub_download(CFG["TS_DATASET"], "TinyStories-train.txt", repo_type="dataset"))
    text = path.read_text(encoding="utf-8")
    return [b.strip() for b in text.split("<|endoftext|>") if b.strip()]


manifest = {"tokenizer": "EleutherAI/gpt-neo-125M", "eos": EOS, "seq_len": SEQ,
            "data_seed": CFG["DATA_SEED"], "max_story_tokens": CFG["MAX_STORY_TOKENS"]}

if CFG["CORPUS"] == "tts":
    stories = tts_stories()
    n = len(stories)
    rng = __import__("random").Random(CFG["DATA_SEED"])
    order = list(range(n))
    rng.shuffle(order)
    val_idx = set(order[:CFG["VAL_STORIES"]])
    train_idx = [i for i in order if i not in val_idx]

    train_info = pack_stories((tokenize_story(stories[i]) for i in train_idx), DATA_DIR / "tts_train.bin")
    val_info = pack_stories((tokenize_story(stories[i]) for i in sorted(val_idx)), DATA_DIR / "tts_val.bin")
    manifest["tts_train"] = train_info
    manifest["tts_val"] = val_info
    print("TTS train:", train_info)
    print("TTS val:", val_info)
else:
    stories = ts_stories()
    n = len(stories)
    rng = __import__("random").Random(CFG["DATA_SEED"])
    order = list(range(n))
    rng.shuffle(order)

    def ts_capped_ids():
        for i in order:
            ids = tokenize_story(stories[i])
            if len(ids) <= CFG["MAX_STORY_TOKENS"]:
                yield ids

    train_info = pack_stories(ts_capped_ids(), DATA_DIR / "ts_train.bin")
    manifest["ts_train"] = train_info
    print("TS capped train:", train_info)

val_path = Path(hf_hub_download(CFG["TS_DATASET"], "TinyStories-valid.txt", repo_type="dataset"))
val_stories = [b.strip() for b in val_path.read_text(encoding="utf-8").split("<|endoftext|>") if b.strip()]
ts_full = pack_stories((tokenize_story(s) for s in val_stories), DATA_DIR / "ts_val_full.bin")


def ts_val_capped_ids():
    for s in val_stories:
        ids = tokenize_story(s)
        if len(ids) <= CFG["MAX_STORY_TOKENS"]:
            yield ids


ts_cap = pack_stories(ts_val_capped_ids(), DATA_DIR / "ts_val_capped.bin")
manifest["ts_val_full"] = ts_full
manifest["ts_val_capped"] = ts_cap
print("real-TS val full:", ts_full)
print("real-TS val capped:", ts_cap)

for info in (train_info, val_info) if CFG["CORPUS"] == "tts" else (train_info,):
    p = Path(info["path"])
    assert p.stat().st_size == info["written_tokens"] * 2, "memmap size mismatch"
(RESULTS / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("wrote results/data_manifest.json")
'''

C6 = '''import contextlib


def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


if CFG["DEVICE"] == "auto":
    DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEV = torch.device(CFG["DEVICE"])
if CFG["AUTOCAST"] == "auto":
    AC_DTYPE = torch.bfloat16 if DEV.type == "cuda" else None
elif CFG["AUTOCAST"] == "fp16":
    AC_DTYPE = torch.float16
else:
    AC_DTYPE = None


def autocast_ctx():
    return contextlib.nullcontext() if AC_DTYPE is None else torch.autocast(DEV.type, dtype=AC_DTYPE)


def empty_cache():
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    elif DEV.type == "mps":
        torch.mps.empty_cache()


def mem_gb():
    if DEV.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if DEV.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


print("device:", DEV, "| autocast:", AC_DTYPE)


def train_arrays():
    if CFG["CORPUS"] == "tts":
        return {"train": load_bin(Path(CFG["DATA_DIR"]) / "tts_train.bin")}
    return {"train": load_bin(Path(CFG["DATA_DIR"]) / "ts_train.bin")}


def eval_arrays():
    d = Path(CFG["DATA_DIR"])
    out = {"ts_val_full": load_bin(d / "ts_val_full.bin"), "ts_val_capped": load_bin(d / "ts_val_capped.bin")}
    if CFG["CORPUS"] == "tts":
        out["tts_val"] = load_bin(d / "tts_val.bin")
    return out


def batch_from(arr, step, micro_batch, seq_len, seed, epoch=0):
    rng = np.random.default_rng([seed, epoch, step])
    hi = len(arr) - seq_len - 1
    assert hi > 0
    ix = rng.integers(0, hi, size=micro_batch)
    x = np.stack([np.asarray(arr[i:i + seq_len], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(arr[i + 1:i + 1 + seq_len], dtype=np.int64) for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


ARRAYS = train_arrays()
EVALS = eval_arrays()
print({k: len(v) for k, v in ARRAYS.items()}, {k: len(v) for k, v in EVALS.items()})
'''

C7 = '''tokens_per_step = CFG["MICRO_BATCH"] * CFG["SEQ_LEN"] * GRAD_ACCUM
assert tokens_per_step == CFG["TOKENS_PER_STEP"]
train_tokens = len(ARRAYS["train"])
own_steps = train_tokens // tokens_per_step
tts_steps = CFG["TTS_TRAIN_TOKENS_WITH_EOS"] // tokens_per_step
if CFG["MATCH_MODE"] == "equal_steps":
    target_steps = tts_steps
else:
    target_steps = own_steps
epochs = target_steps * tokens_per_step / train_tokens
warmup = max(1, int(CFG["WARMUP_FRAC"] * target_steps))
ckpts = target_steps // CFG["CKPT_EVERY_STEPS"] + 1
plan = {"run_id": CFG["RUN_ID"], "corpus": CFG["CORPUS"], "match_mode": CFG["MATCH_MODE"],
        "tokens_per_step": tokens_per_step, "own_steps": own_steps, "tts_reference_steps": tts_steps,
        "target_steps": target_steps, "epochs": round(epochs, 3), "warmup_steps": warmup,
        "eval_every": CFG["EVAL_EVERY_STEPS"], "ckpt_every": CFG["CKPT_EVERY_STEPS"],
        "expected_checkpoints": ckpts, "train_tokens": train_tokens,
        "repeated_fraction": round(max(0.0, epochs - 1.0), 3)}
(Path(CFG["RESULTS_DIR"]) / "planned_params.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
print(json.dumps(plan, indent=2))
print("Review this, then run the training cell.")
'''

C8 = '''if not CFG["RUN_BENCHMARK"]:
    print("benchmark disabled (CFG['RUN_BENCHMARK']=False); set True to measure micro-batch candidates")
else:
    import time
    results = []
    for mb in (4, 8, 16, 32):
        for compile_flag in (False, True):
            if compile_flag and DEV.type != "cuda":
                continue
            try:
                empty_cache()
                if DEV.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                m = model.to(DEV)
                m.train()
                if compile_flag:
                    m = torch.compile(m)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
                g = torch.Generator().manual_seed(0)
                x = torch.randint(0, 50257, (mb, CFG["SEQ_LEN"]), generator=g).to(DEV)
                y = torch.randint(0, 50257, (mb, CFG["SEQ_LEN"]), generator=g).to(DEV)
                for _ in range(3):
                    with autocast_ctx():
                        loss = m(input_ids=x, labels=y).loss
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if DEV.type == "cuda":
                    torch.cuda.synchronize()
                elif DEV.type == "mps":
                    torch.mps.synchronize()
                t0 = time.time()
                steps = 10
                for _ in range(steps):
                    with autocast_ctx():
                        loss = m(input_ids=x, labels=y).loss
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if DEV.type == "cuda":
                    torch.cuda.synchronize()
                elif DEV.type == "mps":
                    torch.mps.synchronize()
                dt = (time.time() - t0) / steps
                tps = mb * CFG["SEQ_LEN"] / dt
                results.append({"micro_batch": mb, "compile": compile_flag, "seconds_per_step": dt,
                                "tokens_per_s": tps, "peak_gb": mem_gb()})
                print(results[-1])
            except RuntimeError as exc:
                results.append({"micro_batch": mb, "compile": compile_flag, "error": str(exc)[:200]})
                print("failed:", results[-1])
    (Path(CFG["RESULTS_DIR"]) / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote results/benchmark.json")
'''

C9 = '''if not CFG["RUN_LR_SWEEP"]:
    print("LR sweep disabled (CFG['RUN_LR_SWEEP']=False); TS-only, short runs, not run by default")
else:
    for lr in (5e-4, 1e-3, 2e-3, 3e-3):
        print(f"LR sweep placeholder for lr={lr}: run the TS arm for a short budget and compare real-TS val loss")
    print("This cell is intentionally a stub: implement after the first test run is reviewed.")
'''

C10 = '''import csv, math, os, random, time

model.to(DEV)
model.train()

steps = plan["target_steps"]
tokens_per_step = plan["tokens_per_step"]
warmup = plan["warmup_steps"]
accum = GRAD_ACCUM
mb = CFG["MICRO_BATCH"]

decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.dim() >= 2:
        decay.append(p)
    else:
        no_decay.append(p)
optim = torch.optim.AdamW([{"params": decay, "weight_decay": CFG["WEIGHT_DECAY"]},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=CFG["LR"], betas=CFG["BETAS"], eps=CFG["EPS"], fused=(DEV.type == "cuda"))

start_step = 0
rng = random.Random(CFG["INIT_SEED"])
if CFG["RESUME_PATH"]:
    ckpt = torch.load(Path(CFG["RESUME_PATH"]) / "state.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    start_step = ckpt["step"]
    rng.setstate(ckpt["rng"])
    torch.set_rng_state(ckpt["torch_rng"])
    print("resumed at step", start_step)


def lr_at(step):
    if step < warmup:
        return CFG["LR"] * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return CFG["LR"] * (CFG["MIN_LR_FRAC"] + (1 - CFG["MIN_LR_FRAC"]) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(arr, max_tokens=None):
    model.eval()
    total_loss, total = 0.0, 0
    limit = len(arr) if max_tokens is None else min(len(arr), max_tokens)
    n_blocks = (limit - 1) // CFG["SEQ_LEN"]
    for b in range(0, n_blocks, mb):
        idx = list(range(b, min(b + mb, n_blocks)))
        x = torch.stack([torch.from_numpy(np.asarray(arr[i * CFG["SEQ_LEN"]:(i + 1) * CFG["SEQ_LEN"]], dtype=np.int64)) for i in idx]).to(DEV)
        y = torch.stack([torch.from_numpy(np.asarray(arr[i * CFG["SEQ_LEN"] + 1:(i + 1) * CFG["SEQ_LEN"] + 1], dtype=np.int64)) for i in idx]).to(DEV)
        with autocast_ctx():
            out = model(input_ids=x, labels=y)
        total_loss += float(out.loss) * x.numel()
        total += x.numel()
    model.train()
    mean = total_loss / max(total, 1)
    return mean, math.exp(mean)


PROMPTS = ["Once upon a time, there was a little", "One day, a boy named Tim found a", "The little cat looked at the big"]


@torch.no_grad()
def sample_prompts():
    model.eval()
    outs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(DEV)
        gen = model.generate(**ids, do_sample=False, max_new_tokens=40, pad_token_id=EOS)
        outs.append(tok.decode(gen[0], skip_special_tokens=True))
    model.train()
    return outs


CKPT_DIR = Path(CFG["CKPT_DIR"])
CKPT_DIR.mkdir(parents=True, exist_ok=True)
log_path = Path(CFG["RESULTS_DIR"]) / "train_log.csv"
log_file = log_path.open("a", newline="")
writer = csv.writer(log_file)
if log_path.stat().st_size == 0:
    writer.writerow(["step", "tokens_seen", "train_loss", "lr", "grad_norm", "tokens_per_s", "gpu_mem_gb"])

t0 = time.time()
running = []
for step in range(start_step, steps):
    lr = lr_at(step)
    for g in optim.param_groups:
        g["lr"] = lr
    optim.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for _ in range(accum):
        x, y = batch_from(ARRAYS["train"], step, mb, CFG["SEQ_LEN"], CFG["DATA_SEED"])
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        with autocast_ctx():
            out = model(input_ids=x, labels=y)
        loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        (loss / accum).backward()
        loss_sum += float(loss.detach()) / accum
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["GRAD_CLIP"])
    optim.step()
    running.append(loss_sum)

    if (step + 1) % 20 == 0 or step == start_step:
        dt = time.time() - t0
        tps = (step + 1 - start_step) * tokens_per_step / dt
        writer.writerow([step + 1, (step + 1) * tokens_per_step, sum(running) / len(running), lr,
                         float(gnorm), round(tps, 1), round(mem_gb(), 2)])
        log_file.flush()
        print(f"step {step + 1}/{steps} | loss {sum(running) / len(running):.4f} | lr {lr:.2e} | {tps:.0f} tok/s")
        running = []

    if (step + 1) % CFG["EVAL_EVERY_STEPS"] == 0:
        for name, arr in EVALS.items():
            mean, ppl = evaluate(arr, CFG["EVAL_SUBSET_TOKENS"])
            print(f"eval step {step + 1} | {name}: loss {mean:.4f} | ppl {ppl:.2f}")
        for s in sample_prompts():
            print("sample:", s[:120])

    if (step + 1) % CFG["CKPT_EVERY_STEPS"] == 0 or (step + 1) == steps:
        ck = CKPT_DIR / f"step{step + 1:07d}"
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step + 1,
                    "rng": rng.getstate(), "torch_rng": torch.get_rng_state()}, ck / "state.pt")
        old = sorted(CKPT_DIR.glob("step*"))
        for o in old[:-CFG["KEEP_LAST_K"]]:
            for f in o.iterdir():
                f.unlink()
            o.rmdir()
        print("checkpoint", ck)

log_file.close()
print("training cell finished")
'''

C11 = '''full = {}
for name, arr in EVALS.items():
    mean, ppl = evaluate(arr)
    full[name] = {"loss_nats": mean, "ppl": ppl, "tokens": len(arr)}
    print(f"final {name}: loss {mean:.4f} | ppl {ppl:.2f}")

model_dir = Path(CFG["RESULTS_DIR"]) / "final_model"
model.save_pretrained(model_dir)
tok.save_pretrained(model_dir)

summary = {"run_id": CFG["RUN_ID"], "corpus": CFG["CORPUS"], "match_mode": CFG["MATCH_MODE"],
           "steps": plan["target_steps"], "tokens_seen": plan["target_steps"] * plan["tokens_per_step"],
           "eval": full, "model_dir": str(model_dir), "seeds": {"init": CFG["INIT_SEED"], "data": CFG["DATA_SEED"], "eval": CFG["EVAL_SEED"]}}
(Path(CFG["RESULTS_DIR"]) / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(Path(CFG["RESULTS_DIR"]) / "training_params.json").write_text(json.dumps({**{k: v for k, v in CFG.items()},
                                                                             "grad_accum": GRAD_ACCUM, "plan": plan}, indent=2, default=str), encoding="utf-8")
print("wrote results/run_summary.json and results/training_params.json")
'''

C12 = '''import importlib, sys
sys.path.insert(0, str(PROJECT_ROOT))
import make_checklist
importlib.reload(make_checklist)
make_checklist.generate()
'''

cells = [
    md(MD0),
    md("## 1. Environment (cheap)\n\nChecks dependencies, accelerator (CUDA or Apple MPS), free disk; writes `results/env.json`. Install commands are shown as a comment if anything is missing."),
    code(C1),
    md("## 2. CONFIG (cheap)\n\nOne `CFG` dict; every line carries its provenance tag and reason. `CORPUS` switches the matched TS arm."),
    code(C2),
    md("## 3. Input verification (moderate: hashes ~4.5 GB)\n\nLists files, sizes, SHA-256; asserts story counts against `results/corpus_stats.json` (TTS) or `results/baseline_ts.json` (TS)."),
    code(C3),
    md("## 4. Model (cheap)\n\nFetches the 1M config from the Hub, saves the raw JSON, asserts the architecture, builds a randomly initialized model, records parameter counts."),
    code(C4),
    md("## 5. Data preparation (HEAVY: tokenizes ~500M tokens)\n\nStream-tokenizes, appends EOS per story, story-level shuffle with `DATA_SEED`, splits off 20,000 validation stories, writes uint16 memmaps, prepares real-TS validation (full and capped), writes `results/data_manifest.json`. Memory note: the TTS branch holds the corpus text in memory (~6-8 GB peak on the 2.5M-story file); the TS branch streams in shuffled index order without accumulating token lists."),
    code(C5),
    md("## 6. Data loader (cheap)\n\nDeterministic random access into the memmaps; batches are derived from the step number, so training is resumable exactly. Train/val never mix."),
    code(C6),
    md("## 7. PLAN (cheap; review before training)\n\nComputes tokens/step, steps, epochs, warmup, eval/checkpoint schedule; writes `results/planned_params.json`."),
    code(C7),
    md("## 8. Throughput benchmark (optional; opt-in)\n\nMeasures tokens/s and peak memory for micro-batch candidates (compile on/off; compile only on CUDA); writes `results/benchmark.json`."),
    code(C8),
    md("## 9. LR sweep (optional; opt-in, TS-only, not run by default)\n\nPlaceholder for short TS runs at {5e-4, 1e-3, 2e-3, 3e-3}; the chosen LR is reused for TTS."),
    code(C9),
    md("## 10. TRAINING (HEAVY)\n\nPlain PyTorch loop: gradient accumulation, device-aware autocast (bf16 on CUDA, fp32 on MPS by default), warmup + cosine, periodic eval on TTS-val and real-TS-val (fixed subsets during training), greedy samples of 3 fixed prompts, NaN abort, checkpoints with exact resume. Logs to `results/train_log.csv`."),
    code(C10),
    md("## 11. Final evaluation and save\n\nFull evaluations on all eval sets; saves the HF-format model and tokenizer; writes `results/run_summary.json` and `results/training_params.json`."),
    code(C11),
    md("## 12. Checklist updater\n\nRegenerates `TRAINING_CHECKLIST.md` from `results/*.json` using the same logic as `make_checklist.py`."),
    code(C12),
]

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

errors = []
for c in cells:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            compile(src, c["id"], "exec")
        except SyntaxError as e:
            errors.append((c["id"], e))
if errors:
    for cid, e in errors:
        print("SYNTAX ERROR in", cid, e)
    raise SystemExit(1)

OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("wrote", OUT, "| cells:", len(cells), "| all code cells compile")

try:
    import nbformat
    nbformat.validate(nbformat.read(OUT, as_version=4))
    print("nbformat validation: OK")
except ImportError:
    print("nbformat not installed; JSON structure validation only")
except Exception as e:
    print("nbformat validation FAILED:", e)
    raise
