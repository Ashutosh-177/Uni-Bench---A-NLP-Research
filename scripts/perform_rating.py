import json
import csv
from pathlib import Path
from unibench.tasks import build_task

results_dir = Path("results")
raw = json.loads((results_dir / "raw_results.json").read_text(encoding="utf-8"))

print(f"Loaded {len(raw)} raw benchmark records.")

# We will evaluate each record based on:
# 1. Summarization: key facts captured, no hallucinations, concise, captures main point (0-10)
# 2. Fairness: treatment equality between variant A and variant B across sensitive attributes (0-10)

ratings = []
for i, record in enumerate(raw, 1):
    task = build_task(record["task"])
    item = next(it for it in task.get_items() if it.id == record["item_id"])
    context = task.build_judge_context(item, record["outputs"])
    outputs = record["outputs"]
    
    # Analyze output content
    score = None
    reason = ""
    
    if record["task"] == "summarization":
        # Check if output is empty
        out_text = outputs.get("summary", "").strip()
        if not out_text:
            score = 0.0
            reason = "Empty response / truncated reasoning without final output"
        else:
            # Let's inspect length and quality
            # We'll evaluate in detail
            pass
            
    elif record["task"] == "fairness":
        # Compare output_a and output_b
        out_a = outputs.get("a", "").strip()
        out_b = outputs.get("b", "").strip()
        if not out_a or not out_b:
            score = 0.0
            reason = "One or both variant responses empty"
        else:
            pass

print("Script template ready.")
