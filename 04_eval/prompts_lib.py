"""Single source of prompts. All scripts must import from here.
Aborts if the frozen file's SHA-256 differs from EXPECTED.
"""
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).parent
PATH = Path("/Users/yanfu/Documents/TeenyTinyStories/cloud_training/prompts_official_style.txt")
EXPECTED = "005759a4cc99efd7ead3460704743b2d056e1ef66f53d1bc0d7c2b04e871d81b"


def load_prompts():
    raw = PATH.read_text(encoding="utf-8")
    h = hashlib.sha256(raw.encode()).hexdigest()
    if h != EXPECTED:
        raise SystemExit(f"prompts_final.txt SHA mismatch: {h} != {EXPECTED}; aborting")
    out = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.*)$", line)
        pid, text = int(m.group(1)), m.group(2)
        text = text.replace("\u201c", '"').replace("\u201d", '"').replace(
            "\u2018", "'").replace("\u2019", "'").rstrip()
        out[pid] = text
    assert sorted(out) == list(range(1, 51)), "need ids 1..50"
    return out
