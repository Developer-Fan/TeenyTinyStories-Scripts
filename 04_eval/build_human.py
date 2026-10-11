"""Build human validation packet: selected_50.csv (stratified, fixed seed),
human_packet.csv (blinded, randomized order), human_form.html, analyze_human.py.
Selection needs only generations (no judge results) so it can be pre-registered.
Usage: python build_human.py
"""
import csv
import html
import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs"
SEED = 2026


def load_gens():
    rows = {}
    with open(OUT / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["model"], r["prompt_id"], r["completion_idx"])] = r
    return rows


from prompts_lib import load_prompts


CRITERIA = ("Decide which story completion is better considering grammar, creativity, "
            "consistency with the beginning, and plot sense. Do not prefer a completion "
            "because of its length or its position. If the two are of genuinely equal "
            "quality, answer TIE.")


def main():
    gens, prompts = load_gens(), load_prompts()
    assert len(gens) == 2000
    models = ["ours_1M", "ours_33M", "ts_1M", "ts_33M"]
    pairs = list(itertools.combinations(models, 2))
    rng = random.Random(SEED)
    # stratify: each pair 8 or 9 times over 50 prompts (50 = 2*8 + 4*9)
    pids = list(range(1, 51))
    rng.shuffle(pids)
    assign = []
    for i, (a, b) in enumerate(pairs):
        assign += [(a, b)] * (9 if i < 2 else 8)
    assert len(assign) == 50  # 2x9 + 4x8
    sel = [{"prompt_id": pid, "model_a": a, "model_b": b}
           for pid, (a, b) in zip(pids, assign)]
    with open(OUT / "selected_50.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "model_a", "model_b"])
        w.writeheader()
        w.writerows(sel)
    # blinded packet: randomize A/B slot per item (fixed seed), record mapping
    rng2 = random.Random(SEED + 1)
    packet = []
    for s in sel:
        pid = s["prompt_id"]
        ta, tb = gens[(s["model_a"], pid, 0)]["text"], gens[(s["model_b"], pid, 0)]["text"]
        flip = rng2.random() < 0.5
        first, second = (tb, ta) if flip else (ta, tb)
        packet.append({"prompt_id": pid, "completion_first": first,
                       "completion_second": second,
                       "first_is": s["model_b"] if flip else s["model_a"],
                       "second_is": s["model_a"] if flip else s["model_b"],
                       "prompt": prompts[pid]})
    with open(OUT / "human_packet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "prompt", "completion_first",
                                          "completion_second", "first_is", "second_is"])
        w.writeheader()
        w.writerows(packet)
    # HTML form
    items = []
    for i, p in enumerate(packet):
        items.append(f"""<div class="item"><h3>Item {i+1} / 50 (prompt {p['prompt_id']})</h3>
<p><b>Beginning:</b> {html.escape(p['prompt'])}</p>
<p><b>Completion A:</b> {html.escape(p['completion_first'])}</p>
<p><b>Completion B:</b> {html.escape(p['completion_second'])}</p>
<p>{html.escape(CRITERIA)}</p>
<label><input type="radio" name="q{i}" value="A"> A</label>
<label><input type="radio" name="q{i}" value="B"> B</label>
<label><input type="radio" name="q{i}" value="TIE"> TIE</label><br>
<label>Comment (optional): <input type="text" name="c{i}" size="80"></label></div><hr>""")
    pids_js = "[" + ",".join(str(p["prompt_id"]) for p in packet) + "]"
    (OUT / "human_form.html").write_text(
        "<html><head><meta charset='utf-8'><title>Story completion rating</title></head>"
        "<body><h2>Story completion rating (blinded)</h2>"
        "<p>Rater id: <input type='text' id='rater'></p>" + "\n".join(items) +
        "<button onclick='save()'>Download ratings CSV</button><script>\n"
        f"const PIDS={pids_js};\n"
        "function save(){let rows=['rater_id,item,prompt_id,verdict,comment'];"
        "let r=(document.getElementById('rater').value||'anon').replaceAll(',',';');"
        "for(let i=0;i<50;i++){let v='';document.getElementsByName('q'+i).forEach(e=>{if(e.checked)v=e.value});"
        "let c=document.getElementsByName('c'+i)[0].value.replaceAll(',',';').replaceAll('\\n',' ');"
        "rows.push(r+','+i+','+PIDS[i]+','+v+','+c);}"
        "let blob=new Blob([rows.join('\\n')],{type:'text/csv'});"
        "let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_ratings_'+r+'.csv';a.click();}"
        "</script></body></html>", encoding="utf-8")
    print("wrote selected_50.csv human_packet.csv human_form.html")


if __name__ == "__main__":
    main()
