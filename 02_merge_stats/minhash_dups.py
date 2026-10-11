#!/usr/bin/env python
"""MinHash near-duplicate analysis for the TeenyTinyStories corpus.

Streams a JSONL corpus (one record per line with a "story" field), computes a
128-permutation MinHash sketch of word 5-gram shingles, LSH-bands the sketches
(16 bands x 8 rows), collects candidate pairs, and verifies exact Jaccard
similarity on the candidates. Deterministic: word ids and hash constants are
fixed; no reliance on PYTHONHASHSEED.
"""

import json
import re
import sys
import time
import os
from collections import defaultdict

import numpy as np

NGRAM = 5
NUM_PERM = 128
BANDS = 16
ROWS = NUM_PERM // BANDS
THRESH = 0.8
MASK = (1 << 64) - 1

FNV = 0x100000001B3
FOLD = np.uint64(0x9E3779B97F4A7C15)
C1 = np.uint64(0xBF58476D1CE4E5B9)
C2 = np.uint64(0x94D049BB133111EB)
S30 = np.uint64(30)
S27 = np.uint64(27)
S31 = np.uint64(31)

WORD = re.compile(r"[a-z']+")


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


class MinHasher:
    def __init__(self):
        self.seeds = np.random.RandomState(20261003).randint(
            0, 1 << 63, size=NUM_PERM
        ).astype(np.uint64)
        self.word_ids = {}

    def wid(self, w):
        i = self.word_ids.get(w)
        if i is None:
            i = len(self.word_ids) + 1
            self.word_ids[w] = i
        return i

    def shingles(self, words):
        n = len(words)
        W = np.fromiter((self.wid(w) for w in words), dtype=np.uint64, count=n)
        if n < NGRAM:
            h = np.uint64(0)
            for x in W:
                h = h * np.uint64(FNV) + x
            return np.array([h], dtype=np.uint64)
        H = np.zeros(n - NGRAM + 1, dtype=np.uint64)
        for k in range(NGRAM):
            H = H * np.uint64(FNV) + W[k:n - NGRAM + 1 + k]
        return H

    def signature(self, H):
        m = H.shape[0]
        buf = np.empty((m, NUM_PERM), dtype=np.uint64)
        np.add(H[:, None], self.seeds[None, :], out=buf)
        buf ^= buf >> S30
        buf *= C1
        buf ^= buf >> S27
        buf *= C2
        buf ^= buf >> S31
        return buf.min(axis=0)

    def band_keys(self, sig):
        M = sig.reshape(BANDS, ROWS)
        acc = M[:, 0].copy()
        for r in range(1, ROWS):
            acc *= FOLD
            acc += M[:, r]
        return acc


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def main():
    path = sys.argv[1]
    out_path = sys.argv[2]

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    n_lines = 0
    with open(path, "rb") as f:
        for _ in f:
            n_lines += 1
    log(f"counted {n_lines:,} records in {path}")

    mh = MinHasher()
    band_store = np.empty((BANDS, n_lines), dtype=np.uint64)
    n_tokens = 0
    n_words = 0
    skipped = 0

    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            text = r.get("story", "")
            n_tokens += int(r.get("n_tokens", 0) or 0)
            words = WORD.findall(norm(text))
            if len(words) < NGRAM:
                skipped += 1
                band_store[:, i] = np.uint64(i + 1) * FOLD  # unique keys, no matches
                continue
            n_words += len(words)
            H = mh.shingles(words)
            sig = mh.signature(H)
            band_store[:, i] = mh.band_keys(sig)
            if (i + 1) % 50000 == 0:
                el = time.time() - t0
                log(f"pass1 {i + 1:,}/{n_lines:,} docs | {n_words:,} words | {el:.0f}s")

    log(f"pass1 done: {n_lines:,} docs | {n_tokens:,} tokens | {n_words:,} words | skipped {skipped}")

    candidate_pairs = set()
    for b in range(BANDS):
        keys = band_store[b]
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        if not (counts > 1).any():
            continue
        idx = np.argsort(inv, kind="stable")
        sorted_inv = inv[idx]
        cuts = np.nonzero(np.diff(sorted_inv))[0] + 1
        groups = np.split(idx, cuts)
        for g in groups:
            if len(g) > 1:
                if len(g) > 500:
                    g = g[:500]
                for a in range(len(g)):
                    for c in range(a + 1, len(g)):
                        candidate_pairs.add((int(g[a]), int(g[c])))
        log(f"band {b}: {len(candidate_pairs):,} candidate pairs so far")

    log(f"candidate pairs: {len(candidate_pairs):,}")

    cand_ids = set()
    for a, b in candidate_pairs:
        cand_ids.add(a)
        cand_ids.add(b)

    shingle_sets = {}
    sigs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in cand_ids:
                words = WORD.findall(norm(json.loads(line).get("story", "")))
                H = mh.shingles(words)
                shingle_sets[i] = set(int(x) for x in H)
                sigs[i] = mh.signature(H)

    verified = []
    for a, b in sorted(candidate_pairs):
        j = jaccard(shingle_sets[a], shingle_sets[b])
        est = float((sigs[a] == sigs[b]).mean())
        if j >= THRESH:
            verified.append((j, est, a, b))
    verified.sort(reverse=True)

    involved = set()
    for _, _, a, b in verified:
        involved.add(a)
        involved.add(b)

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, _, a, b in verified:
        union(a, b)

    comp = defaultdict(int)
    for x in involved:
        comp[find(x)] += 1
    comp_sizes = sorted(comp.values(), reverse=True)

    examples = []
    for j, est, a, b in verified[:20]:
        examples.append({
            "a": a, "b": b,
            "exact_jaccard": round(j, 4),
            "est_jaccard": round(est, 4),
        })

    result = {
        "path": os.path.abspath(path),
        "params": {
            "ngram": NGRAM,
            "num_perm": NUM_PERM,
            "bands": BANDS,
            "rows": ROWS,
            "threshold": THRESH,
            "normalization": "lowercase, collapse whitespace",
            "shingle_unit": "word 5-grams",
        },
        "n_docs": n_lines,
        "n_tokens": n_tokens,
        "n_words": n_words,
        "skipped_docs": skipped,
        "candidate_pairs": len(candidate_pairs),
        "verified_pairs": len(verified),
        "stories_in_near_dup_pairs": len(involved),
        "near_dup_rate": round(len(involved) / n_lines, 6),
        "largest_components": comp_sizes[:20],
        "n_components": len(comp_sizes),
        "examples": examples,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote {out_path}")
    log(json.dumps({k: result[k] for k in (
        "n_docs", "n_tokens", "candidate_pairs", "verified_pairs",
        "stories_in_near_dup_pairs", "near_dup_rate")}, indent=2))


if __name__ == "__main__":
    main()
