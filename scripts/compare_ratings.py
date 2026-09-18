import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import csv
from unibench.tasks import build_task

raw = json.load(open(ROOT / "results" / "raw_results.json", encoding="utf-8"))
primary_map = {(r["model"], r["task"], r["item_id"]): float(r["human_score"]) for r in csv.DictReader(open(ROOT / "results" / "human_ratings.csv", encoding="utf-8"))}
aditya_map = {(r["model"], r["task"], r["item_id"]): float(r["human_score"]) for r in csv.DictReader(open(ROOT / "results" / "human_ratings_aditya.csv", encoding="utf-8"))}

tasks = {}
for i, r in enumerate(raw, 1):
    task = tasks.setdefault(r["task"], build_task(r["task"]))
    item = next(it for it in task.get_items() if it.id == r["item_id"])
    k = (r["model"], r["task"], r["item_id"])
    p_score = primary_map.get(k)
    a_score = aditya_map.get(k)
    print(f"[{i:02d}] {r['model']:15s} | {r['task']:13s} | {r['item_id']:18s} | Primary: {p_score:4.1f} | Aditya: {a_score:4.1f}")
