import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM

SIZES = {
    "1M": {"hidden_size": 64, "num_layers": 8, "num_heads": 16},
    "3M": {"hidden_size": 128, "num_layers": 8, "num_heads": 16},
    "9M": {"hidden_size": 256, "num_layers": 8, "num_heads": 16},
    "28M": {"hidden_size": 512, "num_layers": 8, "num_heads": 16},
    "33M": {"hidden_size": 768, "num_layers": 4, "num_heads": 16},
}
SIZE_NOTES = {
    "1M": "HF roneneldan/TinyStories-1M",
    "3M": "HF roneneldan/TinyStories-3M",
    "9M": "HF roneneldan/TinyStories-8M config (hidden 256); repo TinyStories-9M is inaccessible",
    "28M": "HF roneneldan/TinyStories-28M",
    "33M": "HF roneneldan/TinyStories-33M",
}
TS_DATASET = "roneneldan/TinyStories"
DEFAULT_TTS = "merged/tts_corpus.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Train GPT-Neo students on TinyStories (TS) or TeenyTinyStories (TTS).")
    p.add_argument("--data-kind", choices=["ts", "tts"], required=True)
    p.add_argument("--data-path", default=None,
                   help=f"Local file for training data. Defaults: TS uses HF dataset {TS_DATASET} (train split); TTS uses {DEFAULT_TTS}")
    p.add_argument("--size", choices=sorted(SIZES), default="33M", help="Architecture preset (see --hidden-size to override)")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--max-position-embeddings", type=int, default=512)
    p.add_argument("--activation", default="gelu_new", choices=["gelu", "gelu_new"])
    p.add_argument("--vocab-size", type=int, default=50257)

    p.add_argument("--tokens", type=int, default=50_000_000, help="Training token budget; use the same value for the TS/TTS arms of a pair")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-story-tokens", type=int, default=400, help="Drop stories longer than this (TS filtering to avoid the TTS generation cap confounding)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--scheduler", choices=["cosine", "linear", "constant"], default="cosine")
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-file", default=None, help="Held-out TS file; defaults to the HF validation split")
    p.add_argument("--no-eval", action="store_true")

    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--cache-dir", default="models/cache")
    p.add_argument("--out", default=None, help="Run directory; defaults to models/runs/<kind>_<size>_seed<seed>")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--limit-stories", type=int, default=None, help="Cap stories consumed (smoke tests)")
    return p.parse_args()


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def pick_dtype(name, device):
    if name == "bf16":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("bf16 unavailable; using fp32", flush=True)
        return torch.float32
    if name == "fp16":
        if device in ("cuda", "mps"):
            return torch.float16
        print("fp16 unavailable; using fp32", flush=True)
        return torch.float32
    return torch.float32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_stories_ts(path, limit, log):
    if path:
        text = Path(path).read_text(encoding="utf-8")
        for block in text.split("<|endoftext|>"):
            block = block.strip()
            if block:
                yield block
        return
    from datasets import load_dataset
    log(f"streaming {TS_DATASET} (train)")
    ds = load_dataset(TS_DATASET, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield row["text"]


def iter_stories_tts(path, limit, log):
    p = Path(path)
    log(f"reading {p}")
    if p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    return
                yield json.loads(line)["story"]
        return
    text = p.read_text(encoding="utf-8")
    n = 0
    for block in text.split("<|endoftext|>"):
        block = block.strip()
        if block:
            yield block
            n += 1
            if limit is not None and n >= limit:
                return


def build_token_cache(tok, stories, out_path, tokens, seq_len, max_story_tokens, log):
    tmp = Path(str(out_path) + ".tmp")
    arr = np.memmap(tmp, dtype=np.uint32, mode="w+", shape=(tokens,))
    eos = tok.eos_token_id
    buf = []
    written = 0
    stories_used = 0
    skipped = 0
    t0 = time.time()
    for story in stories:
        ids = tok(story, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            continue
        if len(ids) > max_story_tokens:
            skipped += 1
            continue
        buf.extend(ids)
        buf.append(eos)
        stories_used += 1
        while len(buf) >= seq_len:
            arr[written:written + seq_len] = buf[:seq_len]
            del buf[:seq_len]
            written += seq_len
            if written >= tokens:
                break
        if written >= tokens:
            break
        if stories_used % 20000 == 0:
            log(f"tokenized {stories_used} stories | {written:,}/{tokens:,} tokens | {time.time() - t0:.0f}s")
    arr.flush()
    del arr
    tmp.rename(out_path)
    log(f"cache built: {out_path} | {tokens:,} tokens | {stories_used} stories | skipped {skipped} long stories")
    return stories_used, skipped


def get_tokens(args, tok, log):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = args.data_kind if args.data_path is None else Path(args.data_path).stem
    tokens = (args.tokens // args.seq_len) * args.seq_len
    limit_tag = f"_limit{args.limit_stories}" if args.limit_stories else ""
    cache = cache_dir / f"train_{tag}_{args.data_kind}_maxtok{args.max_story_tokens}_seq{args.seq_len}_{tokens}{limit_tag}.npy"
    meta = cache.with_suffix(".json")
    if cache.exists() and meta.exists():
        log(f"using cached tokens {cache}")
        return np.memmap(cache, dtype=np.uint32, mode="r", shape=(tokens,))
    if args.data_kind == "ts":
        stories = iter_stories_ts(args.data_path, args.limit_stories, log)
    else:
        path = args.data_path or DEFAULT_TTS
        stories = iter_stories_tts(path, args.limit_stories, log)
    used, skipped = build_token_cache(tok, stories, cache, tokens, args.seq_len, args.max_story_tokens, log)
    meta.write_text(json.dumps({"stories_used": used, "skipped_long": skipped, "tokens": tokens}), encoding="utf-8")
    return np.memmap(cache, dtype=np.uint32, mode="r", shape=(tokens,))


def get_eval_tokens(args, tok, log):
    tokens = (args.eval_tokens // args.seq_len) * args.seq_len
    if args.eval_file:
        cache = Path(args.cache_dir) / f"eval_{Path(args.eval_file).stem}_seq{args.seq_len}_{tokens}.npy"
        if cache.exists():
            return np.memmap(cache, dtype=np.uint32, mode="r", shape=(tokens,))
        stories = iter_stories_tts(args.eval_file, None, log)
        build_token_cache(tok, stories, cache, tokens, args.seq_len, 10 ** 9, log)
        return np.memmap(cache, dtype=np.uint32, mode="r", shape=(tokens,))
    from datasets import load_dataset
    cache = Path(args.cache_dir) / f"eval_tsvalid_seq{args.seq_len}_{tokens}.npy"
    if cache.exists():
        log(f"using cached eval tokens {cache}")
        return np.memmap(cache, dtype=np.uint32, mode="r", shape=(tokens,))
    log(f"streaming {TS_DATASET} (validation) for eval")
    ds = load_dataset(TS_DATASET, split="validation", streaming=True)

    def gen():
        for row in ds:
            yield row["text"]

    build_token_cache(tok, gen(), cache, tokens, args.seq_len, 10 ** 9, log)
    return np.memmap(cache, dtype=np.uint32, mode="r", shape=(tokens,))


def lr_at(step, total, args):
    warmup = max(1, int(args.warmup_frac * total))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    if args.scheduler == "constant":
        return args.lr
    progress = (step - warmup) / max(1, total - warmup)
    if args.scheduler == "linear":
        frac = 1.0 - progress
    else:
        frac = args.min_lr_frac + (1.0 - args.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * frac


def get_batch(arr, seq_len, batch_size, device, gen):
    n = len(arr)
    ix = torch.randint(n - seq_len - 1, (batch_size,), generator=gen)
    x = torch.stack([torch.from_numpy(arr[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, arr, args, device):
    model.eval()
    seq_len = args.seq_len
    n_blocks = (len(arr) - 1) // seq_len
    total_loss = 0.0
    total_tokens = 0
    for b in range(0, n_blocks, args.eval_batch_size):
        idx = list(range(b, min(b + args.eval_batch_size, n_blocks)))
        x = torch.stack([torch.from_numpy(arr[i * seq_len:(i + 1) * seq_len].astype(np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(arr[i * seq_len + 1:(i + 1) * seq_len + 1].astype(np.int64)) for i in idx]).to(device)
        out = model(input_ids=x, labels=y)
        total_loss += float(out.loss) * x.numel()
        total_tokens += x.numel()
    model.train()
    mean = total_loss / max(total_tokens, 1)
    return mean, math.exp(mean)


def main():
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    set_seed(args.seed)

    run_dir = Path(args.out) if args.out else Path("models/runs") / f"{args.data_kind}_{args.size}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    t_start = time.time()

    def log(msg):
        line = f"[{time.time() - t_start:8.1f}s] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time() - t_start, 2), "msg": msg}) + "\n")

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", clean_up_tokenization_spaces=False)
    log(f"device={device} dtype={dtype} data={args.data_kind} size={args.size} seed={args.seed}")

    preset = SIZES[args.size]
    hidden = args.hidden_size or preset["hidden_size"]
    layers = args.num_layers or preset["num_layers"]
    heads = args.num_heads or preset["num_heads"]
    assert layers % 2 == 0, "num_layers must be even for GPT-Neo attention_types"
    assert hidden % heads == 0, "hidden_size must be divisible by num_heads"
    config = GPTNeoConfig(
        vocab_size=args.vocab_size,
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        attention_types=[[["global", "local"], layers // 2]],
        window_size=args.window_size,
        max_position_embeddings=args.max_position_embeddings,
        intermediate_size=hidden * 4,
        activation_function=args.activation,
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=False,
    )
    model = GPTNeoForCausalLM(config)
    n_total = sum(p.numel() for p in model.parameters())
    n_nonembed = sum(p.numel() for p in model.parameters()) - config.vocab_size * hidden
    log(f"architecture: hidden={hidden} layers={layers} heads={heads} | params total={n_total:,} non-embedding={n_nonembed:,} | {SIZE_NOTES[args.size]}")
    model.to(device=device, dtype=dtype)
    model.train()

    train_tokens = get_tokens(args, tok, log)
    eval_tokens = None if args.no_eval else get_eval_tokens(args, tok, log)

    steps = args.tokens // (args.batch_size * args.seq_len * args.grad_accum)
    if args.max_steps is not None:
        steps = min(steps, args.max_steps)
    log(f"steps={steps} batch={args.batch_size} accum={args.grad_accum} seq={args.seq_len} lr={args.lr} scheduler={args.scheduler}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=tuple(args.betas), eps=1e-8,
    )
    gen = torch.Generator().manual_seed(args.seed)
    run_meta = {**vars(args), "device": device, "dtype": str(dtype), "hidden_size": hidden, "num_layers": layers,
                "num_heads": heads, "params_total": n_total, "params_non_embedding": n_nonembed, "steps": steps}
    (run_dir / "config.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")

    losses = []
    for step in range(steps):
        lr = lr_at(step, steps, args)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_tokens, args.seq_len, args.batch_size, device, gen)
            out = model(input_ids=x, labels=y)
            loss_val = out.loss.detach().item()
            (out.loss / args.grad_accum).backward()
            acc_loss += loss_val / args.grad_accum
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        losses.append(acc_loss)
        if (step + 1) % args.log_every == 0 or step == 0:
            mean_loss = sum(losses) / len(losses)
            ppl = math.exp(mean_loss)
            log(f"step {step + 1}/{steps} | loss {mean_loss:.4f} | ppl {ppl:.2f} | lr {lr:.2e}")
            losses = []
        if eval_tokens is not None and (step + 1) % args.eval_every == 0:
            mean, ppl = evaluate(model, eval_tokens, args, device)
            log(f"eval step {step + 1} | held-out TS loss {mean:.4f} | ppl {ppl:.2f}")

    if eval_tokens is not None:
        mean, ppl = evaluate(model, eval_tokens, args, device)
        log(f"final eval | held-out TS loss {mean:.4f} | ppl {ppl:.2f}")

    model.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "tokenizer")
    log(f"saved model to {run_dir / 'model'}")


if __name__ == "__main__":
    main()
