"""Headless 20-epoch GPT-Neo training on the TTS corpus for the Vast.ai campaign.

Derived from train_1m_tts.ipynb (cells 0-12). Differences, documented in config.json:
  - EPOCHS=20 with target_steps = (EPOCHS * train_tokens) // TOKENS_PER_STEP
  - distinct micro-batches within each gradient-accumulation step (notebook reused one RNG draw)
  - weight decay excludes embeddings/LayerNorm/biases
  - SDPA attention patch (see sdpa_patch.py), checkpoints every CKPT_EVERY, exact resume
  - interrupt-safe: rolling checkpoints written atomically; SIGTERM saves a checkpoint

Usage:
  python train_cloud.py --size 1m --micro-batch 32 [--resume auto]
"""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM

import sdpa_patch

TOKENIZER = "EleutherAI/gpt-neo-125M"
EOS = 50256
VOCAB = 50257
SEQ_LEN = 512
WINDOW = 256
PROMPTS = ["Once upon a time, there was a little", "One day, a boy named Tim found a",
           "The little cat looked at the big"]

SIZES = {
    "1m": {"hidden": 64, "layers": 8, "heads": 16, "params_total": 3745984,
           "params_non_embedding": 529536, "config_source": "roneneldan/TinyStories-1M", "nominal": "1M"},
    "3m": {"hidden": 128, "layers": 8, "heads": 16, "params_total": 8278400,
           "params_non_embedding": 1845504, "config_source": "roneneldan/TinyStories-3M", "nominal": "3M"},
    "9m": {"hidden": 256, "layers": 8, "heads": 16, "params_total": 19702528,
           "params_non_embedding": 6836736, "config_source": "roneneldan/TinyStories-8M", "nominal": "9M"},
}

PROVENANCE = {
    "epochs": "DECIDED (user, 20)",
    "seq_len": "PROPOSED (512)",
    "tokens_per_step": "PROPOSED (65,536)",
    "micro_batch": "MEASURED (benchmark.json recommendation)",
    "lr": "PROPOSED (1e-3; pending user confirmation)",
    "betas": "PROPOSED (0.9, 0.95)",
    "eps": "PROPOSED (1e-8)",
    "weight_decay": "PROPOSED (0.1, matrices only; embeddings/LayerNorm/bias excluded per prompt)",
    "grad_clip": "PROPOSED (1.0)",
    "warmup_frac": "PROPOSED (0.01)",
    "min_lr_frac": "PROPOSED (0.1)",
    "precision": "PROPOSED (bf16 autocast, fp32 master weights)",
    "init_seed": "PROPOSED (0)",
    "data_seed": "DECIDED (1234; defines split and data order)",
    "eval_seed": "PROPOSED (7)",
    "checkpointing": "PROPOSED (every 2000 steps, keep last 2 + final, exact resume)",
    "architecture": "CONFIG (released roneneldan GPT-Neo configs; 9M = TinyStories-8M, repo TinyStories-9M is 401)",
    "attention": "PROPOSED (F.scaled_dot_product_attention patch, validated against stock eager)",
}


def log(msg, path=None):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path is not None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def build_config(size, device):
    s = SIZES[size]
    config = GPTNeoConfig(
        vocab_size=VOCAB,
        hidden_size=s["hidden"],
        num_layers=s["layers"],
        num_heads=s["heads"],
        attention_types=[[["global", "local"], s["layers"] // 2]],
        window_size=WINDOW,
        max_position_embeddings=2048,
        intermediate_size=4 * s["hidden"],
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=EOS,
        eos_token_id=EOS,
    )
    assert config.num_layers == 8
    assert config.hidden_size == s["hidden"]
    assert config.num_heads == 16
    assert config.window_size == 256
    assert config.vocab_size == 50257
    assert config.attention_types == [[["global", "local"], 4]], config.attention_types
    config._attn_implementation = "eager"
    return config


def param_groups(model, weight_decay):
    excluded = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            for p in module.parameters(recurse=False):
                excluded.add(id(p))
    decay, no_decay = [], []
    for p in model.parameters():
        if id(p) in excluded or p.dim() < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total, warmup, peak, min_frac):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def load_bin(path):
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    assert len(arr) % SEQ_LEN == 0
    return arr


def resolve_micro_batch(args, size):
    if args.micro_batch is not None:
        return args.micro_batch
    bench = Path(args.benchmark_json)
    if bench.exists():
        data = json.loads(bench.read_text(encoding="utf-8"))
        if size in data["recommended"]:
            return int(data["recommended"][size]["micro_batch"])
    raise SystemExit(f"--micro-batch not given and {bench} has no recommendation for {size}; run benchmark.py first")


def load_checkpoint_dir(path):
    path = Path(path)
    state_path = path / "state.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location="cpu", weights_only=False)


def latest_checkpoint(ckpt_root):
    candidates = []
    if not Path(ckpt_root).exists():
        return None
    for d in Path(ckpt_root).glob("step*"):
        if (d / "state.pt").exists():
            try:
                candidates.append((int(d.name[4:]), d))
            except ValueError:
                continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def raw_model(model):
    return getattr(model, "_orig_mod", model)


def save_checkpoint(ckpt_dir, model, optim, step, elapsed, lr):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / "state.pt.tmp"
    state = {
        "model": raw_model(model).state_dict(),
        "optim": optim.state_dict(),
        "scheduler": {"lr": lr, "schedule": "cosine", "scheduler_is_stateless": True},
        "step": step,
        "elapsed_s": elapsed,
        "rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, tmp)
    os.replace(tmp, ckpt_dir / "state.pt")


def prune_checkpoints(ckpt_root, keep_last_k):
    dirs = []
    for d in Path(ckpt_root).glob("step*"):
        if (d / "state.pt").exists():
            try:
                dirs.append((int(d.name[4:]), d))
            except ValueError:
                pass
        else:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    dirs.sort(key=lambda x: x[0])
    for _, d in dirs[:-keep_last_k]:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


@torch.no_grad()
def evaluate(model, arr, micro_batch, max_tokens, device, autocast_dtype):
    model.eval()
    seq = SEQ_LEN
    limit = len(arr) if max_tokens is None else min(len(arr), max_tokens)
    n_blocks = (limit - 1) // seq
    assert n_blocks > 0, f"eval subset too small: {limit} tokens < 2 blocks"
    total_loss, total_tokens = 0.0, 0
    for b in range(0, n_blocks, micro_batch):
        idx = list(range(b, min(b + micro_batch, n_blocks)))
        x = torch.stack([torch.from_numpy(np.asarray(arr[i * seq:(i + 1) * seq], dtype=np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(np.asarray(arr[i * seq + 1:(i + 1) * seq + 1], dtype=np.int64)) for i in idx]).to(device)
        ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
        with ctx:
            out = model(input_ids=x, labels=y)
        total_loss += float(out.loss) * x.numel()
        total_tokens += x.numel()
    model.train()
    mean = total_loss / max(total_tokens, 1)
    return mean, math.exp(mean), total_tokens


@torch.no_grad()
def greedy_samples(model, tok, device):
    model.eval()
    outs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(device)
        gen = model.generate(**ids, do_sample=False, max_new_tokens=40, pad_token_id=EOS)
        outs.append(tok.decode(gen[0], skip_special_tokens=True))
    model.train()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--root", default=str(Path.cwd()))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-manifest", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--benchmark-json", default=None)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--tokens-per-step", type=int, default=65536)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.01)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-subset-tokens", type=int, default=262144)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--keep-last-k", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", default="auto", help="auto | none | path")
    ap.add_argument("--ignore-complete", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-sdpa", action="store_true")
    ap.add_argument("--compile", choices=["none", "default", "max-autotune-no-cudagraphs"], default="none")
    ap.add_argument("--skip-final-eval", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    run_id = args.run_id or f"{args.size.upper()}_tts_seed{args.init_seed}"
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    manifest_path = Path(args.data_manifest) if args.data_manifest else data_dir / "data_manifest.json"
    run_dir = Path(args.run_dir) if args.run_dir else root / "runs" / run_id
    bench_path = Path(args.benchmark_json) if args.benchmark_json else root / "benchmark.json"
    args.benchmark_json = str(bench_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    t_start = time.time()

    if not manifest_path.exists():
        raise SystemExit(f"missing data manifest {manifest_path}; run prepare_data.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_paths = {k: Path(manifest[k]["path"]) for k in ("tts_train", "tts_val", "ts_val_full", "ts_val_capped")}
    for k, p in data_paths.items():
        if not p.is_absolute():
            data_paths[k] = data_dir / p.name

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    micro_batch = resolve_micro_batch(args, args.size)
    seq = args.seq_len
    assert args.tokens_per_step % (micro_batch * seq) == 0, "tokens/step must divide micro_batch*seq"
    grad_accum = args.tokens_per_step // (micro_batch * seq)

    if not args.no_sdpa:
        sdpa_patch.apply()
    log(f"device={device} autocast={autocast_dtype} sdpa={not args.no_sdpa} micro_batch={micro_batch} grad_accum={grad_accum}",
        log_file)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, clean_up_tokenization_spaces=False)
    assert tok.eos_token_id == EOS

    train_arr = load_bin(data_paths["tts_train"])
    evals = {"tts_val": load_bin(data_paths["tts_val"]),
             "ts_val_capped": load_bin(data_paths["ts_val_capped"]),
             "ts_val_full": load_bin(data_paths["ts_val_full"])}
    train_tokens = len(train_arr)
    target_steps = (args.epochs * train_tokens) // args.tokens_per_step
    if args.max_steps is not None:
        target_steps = min(target_steps, args.max_steps)
    assert target_steps > 0
    tokens_seen_nominal = args.epochs * manifest["tts_train"]["counted_tokens"]
    warmup = max(1, int(args.warmup_frac * target_steps))

    config = build_config(args.size, device)
    torch.manual_seed(args.init_seed)
    np.random.seed(args.init_seed)
    random.seed(args.init_seed)
    base_model = GPTNeoForCausalLM(config)
    total_params = sum(p.numel() for p in base_model.parameters())
    emb_params = config.vocab_size * config.hidden_size
    assert total_params == SIZES[args.size]["params_total"], (total_params, SIZES[args.size]["params_total"])
    assert total_params - emb_params == SIZES[args.size]["params_non_embedding"]
    base_model.to(device)
    base_model.train()

    decay_groups = param_groups(base_model, args.weight_decay)
    fused = device.type == "cuda"
    optim = torch.optim.AdamW(decay_groups, lr=args.lr, betas=tuple(args.betas), eps=args.eps, fused=fused)

    model = base_model
    if args.compile != "none" and device.type == "cuda":
        compile_kwargs = {} if args.compile == "default" else {"mode": args.compile}
        model = torch.compile(base_model, **compile_kwargs)
        log(f"torch.compile enabled: {args.compile}", log_file)
    elif args.compile != "none":
        log(f"torch.compile skipped: device={device.type}", log_file)
        args.compile = "none"

    ckpt_root = run_dir / "checkpoints"
    start_step = 0
    elapsed_base = 0.0
    if args.resume != "none":
        resume_dir = None
        if args.resume == "auto":
            if (run_dir / "run_summary.json").exists() and not args.ignore_complete:
                log(f"{run_id} already complete (run_summary.json exists); nothing to do", log_file)
                return
            resume_dir = latest_checkpoint(ckpt_root)
        else:
            resume_dir = Path(args.resume)
        if resume_dir is not None:
            state = load_checkpoint_dir(resume_dir)
            if state is not None:
                base_model.load_state_dict(state["model"])
                optim.load_state_dict(state["optim"])
                start_step = int(state["step"])
                elapsed_base = float(state.get("elapsed_s", 0.0))
                random.setstate(state["rng"])
                torch.set_rng_state(state["torch_rng"])
                if state.get("cuda_rng") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
                log(f"resumed from {resume_dir} at step {start_step}", log_file)

    plan = {
        "run_id": run_id,
        "size": args.size,
        "nominal": SIZES[args.size]["nominal"],
        "config_source": SIZES[args.size]["config_source"],
        "corpus": "tts",
        "epochs": args.epochs,
        "seq_len": seq,
        "tokens_per_step": args.tokens_per_step,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "target_steps": target_steps,
        "train_memmap_tokens": train_tokens,
        "train_counted_tokens_with_eos": manifest["tts_train"]["counted_tokens"],
        "tokens_seen_actual": target_steps * args.tokens_per_step,
        "tokens_seen_nominal_20x_counted": tokens_seen_nominal,
        "epochs_effective": target_steps * args.tokens_per_step / train_tokens,
        "warmup_steps": warmup,
        "params_total": total_params,
        "params_non_embedding": total_params - emb_params,
        "checkpoint_every": args.ckpt_every,
        "eval_every": args.eval_every,
        "data_seed": args.data_seed,
        "init_seed": args.init_seed,
        "eval_seed": args.eval_seed,
        "device": str(device),
        "autocast": str(autocast_dtype),
        "sdpa_patch": not args.no_sdpa,
        "torch_compile": args.compile,
        "note": "epochs are token-budget passes; batches drawn by deterministic random access (seeded by DATA_SEED, epoch, step)",
        "provenance": PROVENANCE,
    }
    (run_dir / "config.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log(json.dumps({k: plan[k] for k in ("target_steps", "tokens_seen_actual", "epochs_effective", "warmup_steps",
                                         "params_total", "params_non_embedding")}, indent=2), log_file)
    if start_step >= target_steps:
        log(f"checkpoint step {start_step} >= target {target_steps}; running finalization only", log_file)

    log_path = run_dir / "train_log.csv"
    log_new = not log_path.exists()
    log_fh = log_path.open("a", newline="")
    writer = csv.writer(log_fh)
    if log_new:
        writer.writerow(["step", "tokens_seen", "train_loss", "lr", "grad_norm", "tokens_per_s", "gpu_mem_gb",
                         "epoch", "elapsed_s"])
    eval_path = run_dir / "eval_log.csv"
    eval_new = not eval_path.exists()
    eval_fh = eval_path.open("a", newline="")
    eval_writer = csv.writer(eval_fh)
    if eval_new:
        eval_writer.writerow(["step", "tokens_seen", "epoch", "dataset", "subset_tokens", "loss_nats", "ppl"])

    samples_path = run_dir / "samples.jsonl"
    done_eval_steps = set()
    if eval_path.exists():
        with eval_path.open() as f:
            next(f, None)
            for row in csv.reader(f):
                if row:
                    done_eval_steps.add(int(row[0]))
    done_sample_steps = set()
    if samples_path.exists():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            try:
                done_sample_steps.add(int(json.loads(line)["step"]))
            except Exception:
                pass
    stop_requested = {"flag": False}

    def on_signal(signum, frame):
        stop_requested["flag"] = True
        log(f"signal {signum} received; will checkpoint and exit after the current step", log_file)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if device.type == "cuda":
        x_step = torch.empty((grad_accum, micro_batch, seq), dtype=torch.long, pin_memory=True)
        y_step = torch.empty_like(x_step)
    else:
        x_step = y_step = None
    running = []
    last_log_step = start_step
    last_log_time = time.time()
    last_ckpt = start_step
    interval_loss_sum = 0.0
    interval_count = 0
    best_step = start_step

    for step in range(start_step, target_steps):
        lr = lr_at(step, target_steps, warmup, args.lr, args.min_lr_frac)
        for g in optim.param_groups:
            g["lr"] = lr
        epoch = min(args.epochs - 1, (step * args.epochs) // target_steps)
        rng = np.random.default_rng([args.data_seed, epoch, step])
        hi = train_tokens - seq - 1
        ix = rng.integers(0, hi, size=(grad_accum, micro_batch))
        xs = np.empty((grad_accum, micro_batch, seq), dtype=np.int64)
        ys = np.empty_like(xs)
        for j in range(grad_accum):
            for k, i in enumerate(ix[j]):
                xs[j, k] = train_arr[i:i + seq]
                ys[j, k] = train_arr[i + 1:i + 1 + seq]
        if device.type == "cuda":
            x_step.copy_(torch.from_numpy(xs))
            y_step.copy_(torch.from_numpy(ys))
        else:
            x_step, y_step = torch.from_numpy(xs), torch.from_numpy(ys)
        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for j in range(grad_accum):
            x = x_step[j].to(device, non_blocking=True)
            y = y_step[j].to(device, non_blocking=True)
            ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
            with ctx:
                out = model(input_ids=x, labels=y)
            loss = out.loss
            if not torch.isfinite(loss):
                save_checkpoint(ckpt_root / "crash", model, optim, step, elapsed_base + time.time() - t_start, lr)
                log(f"NON-FINITE LOSS at step {step}; checkpoint saved at checkpoints/crash", log_file)
                raise SystemExit(3)
            (loss / grad_accum).backward()
            loss_sum += float(loss.detach()) / grad_accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        running.append(loss_sum)
        interval_loss_sum += loss_sum
        interval_count += 1

        if (step + 1) % args.log_every == 0 or step == start_step:
            now = time.time()
            dt = now - last_log_time
            tps = (step + 1 - last_log_step) * args.tokens_per_step / dt if dt > 0 else 0.0
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            writer.writerow([step + 1, (step + 1) * args.tokens_per_step, sum(running) / len(running), lr,
                             float(gnorm), round(tps, 1), round(gpu_mem, 2), epoch,
                             round(elapsed_base + now - t_start, 1)])
            log_fh.flush()
            log(f"step {step + 1}/{target_steps} | loss {sum(running) / len(running):.4f} | lr {lr:.2e} | {tps:.0f} tok/s",
                log_file)
            running = []
            last_log_step = step + 1
            last_log_time = now

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == target_steps or stop_requested["flag"]:
            save_checkpoint(ckpt_root / f"step{step + 1:07d}", model, optim, step + 1,
                            elapsed_base + time.time() - t_start, lr)
            best_step = step + 1
            prune_checkpoints(ckpt_root, args.keep_last_k)
            log(f"checkpoint step {step + 1}", log_file)

        if stop_requested["flag"]:
            log("stopping after checkpoint", log_file)
            log_fh.close()
            eval_fh.close()
            return

        if ((step + 1) % args.eval_every == 0 or (step + 1) == target_steps) and (step + 1) not in done_eval_steps:
            for name, arr in evals.items():
                mean, ppl, ntok = evaluate(model, arr, micro_batch, args.eval_subset_tokens, device, autocast_dtype)
                eval_writer.writerow([step + 1, (step + 1) * args.tokens_per_step, epoch, name, ntok,
                                      round(mean, 6), round(ppl, 4)])
                eval_fh.flush()
                log(f"eval step {step + 1} | {name}: loss {mean:.4f} | ppl {ppl:.2f}", log_file)
            if (step + 1) not in done_sample_steps:
                with samples_path.open("a", encoding="utf-8") as sf:
                    for s in greedy_samples(base_model, tok, device):
                        sf.write(json.dumps({"step": step + 1, "tokens_seen": (step + 1) * args.tokens_per_step,
                                             "text": s}) + "\n")

    log_fh.close()
    eval_fh.close()
    elapsed = elapsed_base + time.time() - t_start

    save_checkpoint(ckpt_root / "final", model, optim, target_steps, elapsed, lr_at(target_steps - 1, target_steps,
                                                                                   warmup, args.lr, args.min_lr_frac))

    full_evals = {}
    if not args.skip_final_eval:
        for name, arr in evals.items():
            mean, ppl, ntok = evaluate(model, arr, micro_batch, None, device, autocast_dtype)
            full_evals[name] = {"loss_nats": round(mean, 6), "ppl": round(ppl, 4), "tokens": ntok}
            log(f"final {name}: loss {mean:.4f} | ppl {ppl:.2f} | tokens {ntok}", log_file)

    final_dir = run_dir / "final_model"
    base_model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    import torch as _torch
    import transformers as _transformers
    import numpy as _numpy

    summary = {
        "run_id": run_id,
        "size": args.size,
        "corpus": "tts",
        "epochs": args.epochs,
        "steps": target_steps,
        "tokens_seen": target_steps * args.tokens_per_step,
        "tokens_seen_nominal_20x_counted": tokens_seen_nominal,
        "eval": full_evals,
        "eval_sets": {"tts_val": "TTS held-out 20k stories", "ts_val_capped": "TinyStories-valid <=400 tokens",
                      "ts_val_full": "TinyStories-valid full"},
        "model_dir": str(final_dir),
        "checkpoints": str(ckpt_root),
        "seeds": {"init": args.init_seed, "data": args.data_seed, "eval": args.eval_seed},
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "tokens_per_step": args.tokens_per_step,
        "torch_compile": args.compile,
        "lr": args.lr,
        "wall_time_s": round(elapsed, 1),
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "torch": _torch.__version__, "transformers": _transformers.__version__,
                     "numpy": _numpy.__version__,
                     "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None},
        "code_sha256": {p: sha256_file(Path(__file__).parent / p) for p in ("train_cloud.py", "sdpa_patch.py")},
        "data_manifest": str(manifest_path),
        "confounds": "nominal 1M/3M/9M assume the paper's top-10K vocab; these runs use the full 50,257 vocab; "
                     "20 epochs on TTS, random-access token-budget epochs; comparison TS checkpoints are released "
                     "pretrained models with unknown schedule and (paper) top-10K vocab",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: {run_id} | steps {target_steps} | wall {elapsed / 3600:.2f} h | model {final_dir}", log_file)


if __name__ == "__main__":
    main()
