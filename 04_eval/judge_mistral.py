"""Mistral judge batch runner (replaces judge_batch.mjs).
Usage: MISTRAL_KEY1=.. MISTRAL_KEY2=.. python judge_mistral.py <jobs> <out> [workers]
Same job/out schema as judge_batch.mjs + {"api": "mistral"}. Skips cached keys.
Round-robins across keys from env MISTRAL_KEY1..N. Retries 4x w/ backoff.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "ministral-14b-latest"
URL = "https://api.mistral.ai/v1/chat/completions"
KEYS = [v for k, v in sorted(os.environ.items()) if k.startswith("MISTRAL_KEY") and v]
assert KEYS, "set MISTRAL_KEY1 (+MISTRAL_KEY2)"
print(f"keys={len(KEYS)} model={MODEL}", flush=True)


def call(messages, key):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("id", "")


def one(job, key):
    att, err = 0, ""
    while att < 4:
        att += 1
        t0 = time.time()
        try:
            txt, mid = call(job["messages"], key)
            return {"key": job["key"], "tag": job["tag"], "response_text": txt,
                    "model_id": f"mistral:{MODEL}", "api": "mistral",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "attempts": att, "error": "", "api_id": mid}
        except Exception as e:
            err = str(e)[:160]
            time.sleep(2 * att)
    return {"key": job["key"], "tag": job["tag"], "response_text": "",
            "model_id": f"mistral:{MODEL}", "api": "mistral",
            "latency_ms": 0, "attempts": att, "error": err}


def main():
    jobs_p, out_p = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    done = set()
    if os.path.exists(out_p):
        with open(out_p, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass
    jobs = [json.loads(l) for l in open(jobs_p, encoding="utf-8") if l.strip()]
    jobs = [j for j in jobs if j["key"] not in done]
    print(f"cached={len(done)} todo={len(jobs)} workers={workers}", flush=True)
    n = [0]
    errs = [0]
    with open(out_p, "a", encoding="utf-8") as fout:
        def run(ix_job):
            ix, job = ix_job
            r = one(job, KEYS[ix % len(KEYS)])
            if r["error"]:
                errs[0] += 1
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            n[0] += 1
            if n[0] % 50 == 0:
                print(f"  {n[0]}/{len(jobs)} errs={errs[0]}", flush=True)
            return r
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, enumerate(jobs)))
    print(f"done errs={errs[0]}", flush=True)


if __name__ == "__main__":
    main()
