#!/usr/bin/env python3
"""train.py: GPT-Neo trainer for the TeenyTinyStories (TTS) vs TinyStories (TS) experiments.


This is project code. It is NOT the TinyStories authors' training code (none has been released as far as we
know). Defaults follow the authors' public statements about the 33M model:
  * model card (roneneldan/TinyStories-33M): LR 5e-4, CONSTANT schedule, weight decay 0.1, Adam betas
    (0.9, 0.95), context length 512;
  * forum reply on that repo's discussion #8: ~20 epochs, 20 sequences per device x 4 GPUs x 16 accumulation
    steps (1,280 sequences, ~655k tokens per step).
That recipe is UNVERIFIED for the other sizes.


Choices that are ours, not the authors':
  * Gradient clipping at 1.0 (the Hugging Face Trainer default; the authors' value is unknown).
  * fp32 throughout (no autocast, TF32 off). The authors trained on V100s, which have no bf16.
  * Weight decay follows the Hugging Face Trainer convention: every parameter is decayed except biases and
    LayerNorm weights, so embeddings ARE decayed. This is an inference (the authors' hyperparameter names look
    like Trainer arguments); --no-decay-embeddings gives the nanoGPT-style alternative.
  * --lr-schedule cosine (with --warmup-frac) is available but is NOT the authors' recipe.
  * Data order: random windows drawn with replacement (the authors' packing/shuffling is unknown).


Design rules:
  * labels == input_ids. Hugging Face shifts labels internally; never pass pre-shifted targets.
  * A startup self-test checks the loss against a hand-written next-token loss, the tied embeddings, token ids
    and (for 1m/3m/9m) the exact parameter count.
  * Samples (greedy and plain T=0.8 sampling, no penalties) are written at every eval. READ THEM early.
  * Atomic checkpoints, SIGTERM-safe, exact resume (a batch depends only on data_seed and the step number).
  * Resuming with a changed recipe is refused unless --allow-config-change is given.
  * "Epochs" are token budgets: windows are drawn at random with replacement, not one strict pass over the data.


Data: uint16 .bin token files (GPT-Neo tokenizer, EOS 50256 after every story).


Smoke test (free, any machine):
  python train.py --size 1m --train data/tts_train.bin --val tts_val=data/tts_val.bin \
      --out runs/smoke --max-steps 30 --eval-every 10 --eval-blocks 8 --micro-batch 8 \
      --tokens-per-step 32768 --device cpu
Real run:
  python train.py --size 1m --train data/tts_train.bin \
      --val tts_val=data/tts_val.bin ts_val_capped=data/ts_val_capped.bin ts_val_full=data/ts_val_full.bin \
      --out runs/1m_tts --micro-batch 32 --eval-every 250 --ckpt-every 500 --keep-epochs 1,2,4,8,12,20
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import sys
import time
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM


SEQ = 512
EOS = 50256
VOCAB = 50257
# (layers, hidden). 1m and 33m were read from released configs. 3m/9m/28m are inferred from the paper's
# figures: VERIFY against roneneldan/TinyStories-3M / -8M (or -9M) / -28M before using them in a paper.
# Head count (16) is hard-coded for all sizes: also verify against the released config.json files.
# Nominal names do not match parameter counts (see plan.json for the measured totals).
PRESETS = {"1m": (8, 64), "3m": (8, 128), "9m": (8, 256), "28m": (8, 512), "33m": (4, 768)}
EXPECTED_PARAMS = {"1m": 3745984, "3m": 8278400, "9m": 19702528}   # checked against instantiated models
PROMPTS = ["Once upon a time, there was a little",
           "One day, a boy named Tim found a",
           "The little cat looked at the big"]
RESUME_KEYS = ("size", "tokens_per_step", "lr", "epochs", "data_seed", "seed", "wd", "betas",
               "warmup_frac", "min_lr_frac", "clip", "lr_schedule", "no_decay_embeddings")




def make_model(size):
    layers, hidden = PRESETS[size]
    cfg = GPTNeoConfig(
        vocab_size=VOCAB, hidden_size=hidden, num_layers=layers, num_heads=16,
        attention_types=[[["global", "local"], layers // 2]], window_size=256,
        max_position_embeddings=2048, intermediate_size=4 * hidden, activation_function="gelu_new",
        resid_dropout=0.0, embed_dropout=0.0, attention_dropout=0.0,
        bos_token_id=EOS, eos_token_id=EOS)
    return GPTNeoForCausalLM(cfg)




def param_groups(model, wd, decay_embeddings=True):
    """Hugging Face Trainer convention: decay everything except biases and LayerNorm weights.
    With decay_embeddings=False the (tied) token and position embeddings are also excluded."""
    ln_ids = {id(p) for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
              for p in m.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        skip = (id(p) in ln_ids or name.endswith("bias")
                or (not decay_embeddings and ("wte" in name or "wpe" in name)))
        (no_decay if skip else decay).append(p)
    assert len(decay) + len(no_decay) == sum(1 for _ in model.parameters())
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]




def lr_at(step, total, warmup, peak, min_frac, schedule):
    if step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "constant":
        return peak
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))




def get_batch(arr, rng, n):
    ix = rng.integers(0, len(arr) - SEQ, size=n)
    return torch.from_numpy(np.stack([arr[i:i + SEQ] for i in ix]).astype(np.int64))




def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)




def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()




@torch.no_grad()
def evaluate(model, arr, max_blocks, mb, dev):
    model.eval()
    n = min(len(arr) // SEQ, max_blocks)
    total = 0.0
    for b in range(0, n, mb):
        idx = range(b, min(b + mb, n))
        x = torch.from_numpy(np.stack([arr[i * SEQ:(i + 1) * SEQ] for i in idx]).astype(np.int64)).to(dev)
        loss = model(input_ids=x, labels=x).loss
        total += loss.item() * x.shape[0] * (SEQ - 1)
    model.train()
    return total / (n * (SEQ - 1)), n * SEQ




@torch.no_grad()
def sample(model, tok, dev):
    model.eval()
    outs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        for kw in (dict(do_sample=False), dict(do_sample=True, temperature=0.8, top_k=0, top_p=1.0)):
            g = model.generate(ids, max_new_tokens=60, pad_token_id=EOS,
                               repetition_penalty=1.0, no_repeat_ngram_size=0, **kw)
            outs.append(tok.decode(g[0], skip_special_tokens=True))
    model.train()
    return outs




def self_test(model, size, train, vals, dev, fresh):
    """HF loss must equal a hand-written next-token loss; a fresh model should score about ln(V)."""
    x = torch.from_numpy(np.stack([train[i * SEQ:(i + 1) * SEQ] for i in range(4)]).astype(np.int64)).to(dev)
    with torch.no_grad():
        out = model(input_ids=x, labels=x)
        lg = out.logits.float()
    manual = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1)).item()
    assert abs(manual - out.loss.item()) < 1e-3, f"loss/target mismatch: manual {manual} vs HF {out.loss.item()}"
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(), "embeddings are not tied"
    for name, arr in [("train", train)] + list(vals.items()):
        assert len(arr) >= 2 * SEQ, f"{name}: file too small"
        assert int(arr[:10_000_000].max()) < VOCAB, f"{name}: token id out of range"
    n_params = sum(p.numel() for p in model.parameters())
    if size in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[size], f"param count {n_params} != expected {EXPECTED_PARAMS[size]}"
    if fresh and abs(out.loss.item() - math.log(VOCAB)) > 0.7:
        print(f"WARNING: initial loss {out.loss.item():.2f}, expected about {math.log(VOCAB):.2f}", flush=True)
    print(f"self-test ok: loss {out.loss.item():.4f} (manual {manual:.4f}), params {n_params:,}", flush=True)
    return n_params




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(PRESETS), required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", nargs="*", default=[], help="name=path.bin ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=20)
    ap.add_argument("--tokens-per-step", type=int, default=655360)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                    help="constant is the authors' recipe (33M model card); cosine is our own option")
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--no-decay-embeddings", action="store_true",
                    help="exclude token/position embeddings from weight decay (default: decay them, "
                         "as the HF Trainer does)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="fraction of steps for linear warmup (default 0 = none, as for a plain constant schedule)")
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="final LR fraction (cosine only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=0, help="also keep a permanent checkpoint every N steps")
    ap.add_argument("--keep-epochs", default="", help="comma list of nominal epochs to keep, e.g. 1,2,4,8,12,20")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-blocks", type=int, default=512, help="blocks per periodic eval (final eval uses all)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hash-data", action="store_true", help="record SHA-256 of the .bin files in plan.json")
    ap.add_argument("--allow-config-change", action="store_true")
    args = ap.parse_args()


    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print("already finished:", out)
        return
    dev = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else torch.device(args.device))
    # strict fp32: no autocast, and TF32 disabled so matmuls are true fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


    train = np.memmap(args.train, dtype=np.uint16, mode="r")
    vals = {}
    for item in args.val:
        name, path = item.split("=", 1)
        vals[name] = np.memmap(path, dtype=np.uint16, mode="r")


    assert args.tokens_per_step % (args.micro_batch * SEQ) == 0, \
        "tokens-per-step must be a multiple of micro-batch*512"
    accum = args.tokens_per_step // (args.micro_batch * SEQ)
    target = int(args.epochs * len(train)) // args.tokens_per_step
    if args.max_steps:
        target = min(target, args.max_steps)
    assert target > 0, "no steps to run (check --epochs and --tokens-per-step)"
    warmup = int(args.warmup_frac * target)
    keep_steps = {int(float(e) * len(train) / args.tokens_per_step) for e in args.keep_epochs.split(",") if e.strip()}
    keep_steps = {s for s in keep_steps if 0 < s <= target}


    torch.manual_seed(args.seed)
    model = make_model(args.size).to(dev)
    model.train()
    optim = torch.optim.AdamW(param_groups(model, args.wd, decay_embeddings=not args.no_decay_embeddings),
                              lr=args.lr, betas=tuple(args.betas), eps=1e-8, fused=(dev.type == "cuda"))
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", clean_up_tokenization_spaces=False)
    assert tok.eos_token_id == EOS


    plan_path, latest = out / "plan.json", out / "latest.pt"
    new_plan = {k: (list(getattr(args, k)) if isinstance(getattr(args, k), tuple) else getattr(args, k))
                for k in RESUME_KEYS}
    start, elapsed_base = 0, 0.0
    if latest.exists():
        if plan_path.exists() and not args.allow_config_change:
            old = json.loads(plan_path.read_text())
            diff = {k: (old.get(k), new_plan[k]) for k in RESUME_KEYS if old.get(k) != new_plan[k]}
            if diff:
                raise SystemExit(f"refusing to resume with a changed recipe (old, new): {diff}. "
                                 f"Use --allow-config-change if this is intended.")
        st = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optim.load_state_dict(st["optim"])
        start, elapsed_base = int(st["step"]), float(st.get("elapsed", 0.0))
        with open(out / "resume_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} resumed at step {start}\n")
        print(f"resumed at step {start}", flush=True)
    n_params = self_test(model, args.size, train, vals, dev, fresh=(start == 0))


    plan = dict(vars(args), params_total=n_params, params_non_embedding=n_params - VOCAB * PRESETS[args.size][1],
                grad_accum=accum, target_steps=target, warmup_steps=warmup, train_tokens=len(train),
                tokens_total=target * args.tokens_per_step,
                epochs_effective=target * args.tokens_per_step / len(train),
                keep_steps=sorted(keep_steps), device=str(dev), torch=torch.__version__, precision="fp32",
                data_files={"train": {"path": args.train, "bytes": os.path.getsize(args.train)},
                            **{n: {"bytes": os.path.getsize(p.split("=")[1])} for n, p in
                               ((i.split("=")[0], i) for i in args.val)}})
    if args.hash_data and start == 0:
        plan["data_files"]["train"]["sha256"] = sha256_file(args.train)
        for item in args.val:
            n, p = item.split("=", 1)
            plan["data_files"][n]["sha256"] = sha256_file(p)
    plan_path.write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: plan[k] for k in ("params_total", "grad_accum", "target_steps", "epochs_effective")}),
          flush=True)


    stop = {"flag": False}


    def on_signal(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)


    t_start = time.time()


    def elapsed():
        return elapsed_base + time.time() - t_start


    def checkpoint(step):
        save_atomic({"model": model.state_dict(), "optim": optim.state_dict(), "step": step,
                     "elapsed": elapsed()}, latest)
        if step in keep_steps or (args.keep_every and step % args.keep_every == 0):
            save_atomic({"model": model.state_dict(), "step": step}, out / f"ckpt_step{step:07d}.pt")


    def run_evals(step, full=False):
        # periodic evals use a fixed subset (first eval-blocks blocks); the final eval covers the whole file
        results = {}
        for name, arr in vals.items():
            loss, ntok = evaluate(model, arr, 10 ** 12 if full else args.eval_blocks, args.micro_batch, dev)
            results[name] = {"loss_nats": round(loss, 6), "ppl": round(math.exp(loss), 4), "tokens": ntok}
            with open(out / "eval_log.csv", "a", newline="") as f:
                csv.writer(f).writerow([step, step * args.tokens_per_step, name, ntok, round(loss, 5),
                                        round(math.exp(loss), 3)])
            print(f"eval step {step} | {name}: loss {loss:.4f} | ppl {math.exp(loss):.2f}", flush=True)
        with open(out / "samples.txt", "a", encoding="utf-8") as f:
            f.write(f"=== step {step} ===\n")
            for s in sample(model, tok, dev):
                f.write(repr(s) + "\n")
        print(open(out / "samples.txt", encoding="utf-8").read().split("=== step")[-1][:900], flush=True)
        return results


    log_new = not (out / "train_log.csv").exists()
    log_f = open(out / "train_log.csv", "a", newline="")
    log_w = csv.writer(log_f)
    if log_new:
        log_w.writerow(["step", "tokens_seen", "epoch", "train_loss", "lr", "grad_norm", "tokens_per_s"])
    if not (out / "eval_log.csv").exists():
        with open(out / "eval_log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens_seen", "dataset", "tokens_evaluated", "loss_nats", "ppl"])
    t_last, step_last, acc_loss, final_eval = time.time(), start, [], {}


    for step in range(start, target):
        lr = lr_at(step, target, warmup, args.lr, args.min_lr_frac, args.lr_schedule)
        for g in optim.param_groups:
            g["lr"] = lr
        rng = np.random.default_rng([args.data_seed, step])
        optim.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            x = get_batch(train, rng, args.micro_batch).to(dev)
            loss = model(input_ids=x, labels=x).loss
            if not torch.isfinite(loss):
                save_atomic({"model": model.state_dict(), "step": step}, out / "crash.pt")
                print(f"NON-FINITE LOSS at step {step}; weights saved to crash.pt", flush=True)
                raise SystemExit(3)
            (loss / accum).backward()
            total += loss.item() / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        optim.step()
        acc_loss.append(total)
        s = step + 1


        if s % args.log_every == 0 or s == 1:
            now = time.time()
            tps = (s - step_last) * args.tokens_per_step / max(now - t_last, 1e-9)
            ep = s * args.tokens_per_step / len(train)
            log_w.writerow([s, s * args.tokens_per_step, round(ep, 4), round(sum(acc_loss) / len(acc_loss), 5),
                            f"{lr:.3e}", round(gnorm, 4), round(tps)])
            log_f.flush()
            print(f"step {s}/{target} | loss {sum(acc_loss) / len(acc_loss):.4f} | lr {lr:.2e} | "
                  f"{tps:.0f} tok/s | epoch {ep:.2f}", flush=True)
            acc_loss, t_last, step_last = [], now, s


        if s % args.ckpt_every == 0 or s == target or s in keep_steps or stop["flag"]:
            checkpoint(s)
        if (s % args.eval_every == 0 or s == target) and vals:
            res = run_evals(s, full=(s == target))
            if s == target:
                final_eval = res
        if stop["flag"]:
            print("stop requested: checkpoint written, exiting", flush=True)
            return


    log_f.close()
    if not final_eval and vals:      # e.g. resumed after the last step but before finalization
        final_eval = run_evals(target, full=True)
    final = out / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    summary = {
        "steps": target, "tokens_seen": target * args.tokens_per_step,
        "epochs_effective": plan["epochs_effective"], "final_eval_full_files": final_eval,
        "wall_time_s": round(elapsed(), 1), "model_dir": str(final),
        "seeds": {"init": args.seed, "data": args.data_seed},
        "recipe": new_plan, "micro_batch": args.micro_batch, "grad_accum": accum,
        "params_total": n_params, "precision": "fp32",
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "numpy": np.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "code_sha256": sha256_file(__file__),
        "note": "epochs are nominal token budgets (random windows with replacement)",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("done:", final, flush=True)




if __name__ == "__main__":
    main()
