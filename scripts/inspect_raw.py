import json
from pathlib import Path

raw = json.load(open("results/raw_results.json", encoding="utf-8"))
for i, r in enumerate(raw):
    outs = r["outputs"]
    has_empty = any(not (v or "").strip() for v in outs.values())
    print(f"{i+1:02d}. {r['model']} | {r['task']} | {r['item_id']} | Empty: {has_empty}")
