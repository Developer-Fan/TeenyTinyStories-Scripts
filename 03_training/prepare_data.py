"""Tokenize TTS train/val and TinyStories-valid full/capped into uint16 memmaps.

Usage (pod):
  python prepare_data.py --root /workspace/cloud_training [--tts-file /path/tts_corpus.txt]

Writes {root}/data/*.bin and {root}/data/data_manifest.json.
Re-run with --force to rebuild.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

TOKENIZER = "EleutherAI/gpt-neo-125M"
EOS = 50256
VOCAB = 50257
SEP = "<|endoftext|>"

TTS_REPO = "ableian/teenytinystories"
TTS_REPO_FILE = "tts_corpus.txt"
TTS_BYTES = 2068689233
TTS_SHA256 = "04dd47ecc160033723e36d707a0766422a39bbb946fcfde804dc7fdc19042fd3"
TTS_STORIES_TOTAL = 2511490

TS_REPO = "roneneldan/TinyStories"
TS_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TS_VALID_FILE = "TinyStories-valid.txt"
TS_VALID_BYTES = 19447282
TS_VALID_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
TS_VALID_STORIES = 21990

EXPECT = {
    "tts_train": {"stories": 2491490, "counted_tokens": 498492043, "written_tokens": 498491904},
    "tts_val": {"stories": 20000, "counted_tokens": 4020250, "written_tokens": 4020224},
    "ts_val_full": {"stories": 21990, "counted_tokens": 4667152, "written_tokens": 4666880},
    "ts_val_capped": {"stories": 20711, "counted_tokens": 3998033, "written_tokens": 3997696},
}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, expected_bytes, expected_sha, label):
    path = Path(path)
    actual_bytes = path.stat().st_size
    assert actual_bytes == expected_bytes, f"{label}: size {actual_bytes} != {expected_bytes}"
    actual_sha = sha256(path)
    assert actual_sha == expected_sha, f"{label}: sha256 {actual_sha} != {expected_sha}"
    log(f"{label}: {actual_bytes:,} bytes | sha256 OK")
    return path


def read_stories(path):
    buf = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(SEP)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    yield p
    tail = buf.strip()
    if tail:
        yield tail


def tokenize_story(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > 0
    assert max(ids) < VOCAB
    return ids


def pack_stories(tok, stories_iter, out_path, seq_len, label, progress_every=200000):
    tmp = Path(str(out_path) + ".tmp")
    buf = []
    block_bytes = bytearray()
    counted = 0
    written = 0
    n_stories = 0
    f = open(tmp, "wb")
    try:
        for text in stories_iter:
            ids = tokenize_story(tok, text)
            buf.extend(ids)
            buf.append(EOS)
            counted += len(ids) + 1
            n_stories += 1
            while len(buf) >= seq_len:
                chunk = np.asarray(buf[:seq_len], dtype=np.uint16)
                block_bytes += chunk.tobytes()
                del buf[:seq_len]
                written += seq_len
                if len(block_bytes) >= (1 << 24):
                    f.write(block_bytes)
                    block_bytes.clear()
            if n_stories % progress_every == 0:
                log(f"{label}: {n_stories:,} stories | {counted:,} tokens")
        if block_bytes:
            f.write(block_bytes)
    finally:
        f.close()
    tmp.replace(out_path)
    info = {"path": str(out_path), "stories": n_stories, "counted_tokens": counted,
            "written_tokens": written, "dropped_tail": counted - written}
    log(f"{label}: {info}")
    return info


def resolve_tts(args):
    candidates = [Path(args.tts_file)] if args.tts_file else [
        Path(args.root) / "data" / TTS_REPO_FILE,
        Path(args.root) / TTS_REPO_FILE,
        Path(args.root) / "merged" / TTS_REPO_FILE,
    ]
    for cand in candidates:
        if cand.exists():
            verify_file(cand, TTS_BYTES, TTS_SHA256, "tts_corpus.txt")
            return {"path": str(cand), "source": "local", "repo": None}
    from huggingface_hub import hf_hub_download

    log(f"downloading {TTS_REPO}/{TTS_REPO_FILE}")
    path = hf_hub_download(TTS_REPO, TTS_REPO_FILE, repo_type="dataset")
    verify_file(path, TTS_BYTES, TTS_SHA256, "tts_corpus.txt")
    return {"path": str(path), "source": "hf", "repo": TTS_REPO}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.cwd()))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tts-file", default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--val-stories", type=int, default=20000)
    ap.add_argument("--max-story-tokens", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "data_manifest.json"
    if manifest_path.exists() and not args.force:
        print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), indent=2)[:4000])
        log("manifest exists; use --force to rebuild")
        return

    tok = AutoTokenizer.from_pretrained(TOKENIZER, clean_up_tokenization_spaces=False)
    assert tok.eos_token_id == EOS, tok.eos_token_id

    tts_src = resolve_tts(args)
    try:
        from huggingface_hub import HfApi

        tts_src["repo_sha"] = HfApi().dataset_info(TTS_REPO).sha
    except Exception as exc:
        tts_src["repo_sha"] = f"PENDING ({str(exc)[:80]})"
    tts_stories = list(read_stories(tts_src["path"]))
    assert len(tts_stories) == TTS_STORIES_TOTAL, len(tts_stories)
    log(f"TTS stories: {len(tts_stories):,}")

    order = list(range(len(tts_stories)))
    random.Random(args.data_seed).shuffle(order)
    val_idx = set(order[: args.val_stories])
    train_idx = [i for i in order if i not in val_idx]
    assert len(train_idx) == len(tts_stories) - args.val_stories

    tts_train = pack_stories(tok, (tts_stories[i] for i in train_idx), data_dir / "tts_train.bin",
                             args.seq_len, "tts_train")
    tts_val = pack_stories(tok, (tts_stories[i] for i in sorted(val_idx)), data_dir / "tts_val.bin",
                           args.seq_len, "tts_val")
    del tts_stories

    from huggingface_hub import hf_hub_download

    valid_path = hf_hub_download(TS_REPO, TS_VALID_FILE, repo_type="dataset", revision=TS_REVISION)
    verify_file(valid_path, TS_VALID_BYTES, TS_VALID_SHA256, "TinyStories-valid.txt")
    valid_stories = list(read_stories(valid_path))
    assert len(valid_stories) == TS_VALID_STORIES, len(valid_stories)
    log(f"TS valid stories: {len(valid_stories):,}")

    ts_val_full = pack_stories(tok, iter(valid_stories), data_dir / "ts_val_full.bin",
                               args.seq_len, "ts_val_full")
    ts_val_capped = pack_stories(tok, (s for s in valid_stories if len(tokenize_story(tok, s)) <= args.max_story_tokens),
                                 data_dir / "ts_val_capped.bin", args.seq_len, "ts_val_capped")

    results = {"tts_train": tts_train, "tts_val": tts_val,
               "ts_val_full": ts_val_full, "ts_val_capped": ts_val_capped}
    for name, info in results.items():
        exp = EXPECT[name]
        for key in ("stories", "counted_tokens", "written_tokens"):
            assert info[key] == exp[key], f"{name}.{key}: {info[key]} != {exp[key]}"
        assert info["written_tokens"] % args.seq_len == 0
        assert 0 <= info["dropped_tail"] < args.seq_len
        arr = np.memmap(info["path"], dtype=np.uint16, mode="r")
        assert len(arr) == info["written_tokens"]
        assert int(arr.max()) < VOCAB, f"{name}: token id >= vocab"
        assert bool((arr == EOS).any()), f"{name}: no EOS found"
        del arr
    log("all token totals and EOS/vocab checks PASSED")

    import platform

    import torch
    import transformers

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokenizer": TOKENIZER,
        "eos": EOS,
        "vocab_size": VOCAB,
        "seq_len": args.seq_len,
        "data_seed": args.data_seed,
        "val_stories": args.val_stories,
        "max_story_tokens": args.max_story_tokens,
        "tts_corpus_source": {**tts_src, "bytes": TTS_BYTES, "sha256": TTS_SHA256,
                              "stories": TTS_STORIES_TOTAL},
        "ts_valid_source": {"path": str(valid_path), "source": "hf", "repo": TS_REPO,
                            "revision": TS_REVISION, "bytes": TS_VALID_BYTES,
                            "sha256": TS_VALID_SHA256, "stories": TS_VALID_STORIES},
        "tts_train": tts_train,
        "tts_val": tts_val,
        "ts_val_full": ts_val_full,
        "ts_val_capped": ts_val_capped,
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "torch": torch.__version__, "transformers": transformers.__version__},
        "provenance": {
            "sizes/sha256": "MEASURED (Mac, MANIFEST.md / results/baseline_ts.json)",
            "token_totals": "MEASURED (Mac, results/data_manifest.json)",
            "split": "DECIDED: DATA_SEED=1234 story shuffle, 20,000-story TTS val",
            "ts_filter": "DECIDED: drop TS valid stories > 400 tokens for the capped set",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
