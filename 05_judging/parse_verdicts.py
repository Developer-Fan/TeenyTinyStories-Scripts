r"""Strict verdict parser and call-log summary for the second-judge run.

The regex and the parsing steps are copied from cloud_training/eval/aggregate_pairwise.py:
  RX = ^\s*Verdict:\s*(A|B|TIE)\s*$  (IGNORECASE | MULTILINE), applied after removing "**",
  taking the LAST match. Responses with no match are FAILED. No free-text inference.

usage: python parse_verdicts.py <calls_log.jsonl> [--phase pilot|full|retry] [--show-failures]
"""
import argparse
import json
import re
from collections import Counter

RX = re.compile(r"^\s*Verdict:\s*(A|B|TIE)\s*$", re.IGNORECASE | re.MULTILINE)


def parse(text):
    """Returns (verdict or None, status, n_distinct_verdict_lines)."""
    if text is None:
        return None, "NO_TEXT", 0
    ms = RX.findall(text.replace("**", ""))
    if not ms:
        return None, "NO_MATCH", 0
    return ms[-1].upper(), "OK", len({m.upper() for m in ms})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--show-failures", action="store_true")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
    if a.phase:
        recs = [r for r in recs if r["phase"] == a.phase]
    att1 = [r for r in recs if r["attempt"] == 1]
    n = len(att1)
    api_err = [r for r in att1 if not r["ok"]]
    parse_fail = []
    verdicts = Counter()
    for r in att1:
        if not r["ok"]:
            continue
        v, status, ndist = parse(r["response_text"])
        if status != "OK":
            parse_fail.append((r, status))
        else:
            verdicts[v] += 1
            if ndist > 1:
                print(f"NOTE conflicting verdict lines: {r['key']}")
    no_verdict = len(api_err) + len(parse_fail)
    print(f"phase={a.phase or 'all'} attempt1 n={n}")
    print(f"  api_errors={len(api_err)}")
    print(f"  parse_failures={len(parse_fail)}  rate={len(parse_fail)/max(n,1):.3f}")
    print(f"  no_verdict_total={no_verdict}  rate={no_verdict/max(n,1):.3f}")
    print(f"  verdicts={dict(verdicts)}")
    print(f"  models_returned={Counter(str(r.get('model_returned')) for r in att1)}")
    print(f"  finish_reasons={Counter(str(r.get('finish_reason')) for r in att1)}")
    if a.show_failures:
        for r in api_err:
            print(f"\n--- API ERROR {r['key']}#{r['attempt']} ---\n{r['error']}")
        for r, status in parse_fail:
            print(f"\n--- PARSE {status} {r['key']}#{r['attempt']} finish={r.get('finish_reason')} ---")
            print(repr(r["response_text"]))


if __name__ == "__main__":
    main()
