"""Re-runs the judging stage of a completed run under the CURRENT pipeline.

The paper's reported tables were produced before three corrections landed, all
of which live in the judging path rather than in generation:

  1. an empty response is marked explicitly (EMPTY_RESPONSE_MARKER),
  2. the reference answer is shown BEFORE the response and labelled as a
     reference, so it cannot be graded in place of an empty one,
  3. an output with an empty response is not sent to the judges at all.

Because none of that touches how the subject models were prompted, the stored
outputs are still valid inputs: re-running judging on them is a real rerun of
the corrected pipeline, not a sensitivity analysis over old verdicts. That
distinction is the difference between "we estimate the correction would have
changed X" and "here is the corrected pipeline's result".

What it cannot recover is a judge that no longer exists. qwen-3.6-27b was
retired by its provider mid-project and 404s, so a corrected-pipeline panel is
necessarily smaller than the one originally reported. That is a real limit and
the report should say so rather than quietly drop it.

Usage:
    python scripts/rejudge.py --judges gpt-oss-120b,gpt-oss-20b
    [--source results] [--out results_corrected] [--config config.paper.yaml]

Writes <out>/raw_results.json with the same subject outputs and telemetry, and
fresh judge verdicts. The source directory is never modified. Resumable: a
record already judged in <out> is left alone.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unibench.evaluation.ai_judge import judge_item  # noqa: E402
from unibench.models import build_client  # noqa: E402
from unibench.tasks import build_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", required=True, help="Comma-separated judge names from the config.")
    ap.add_argument("--source", default="results")
    ap.add_argument("--out", default="results_corrected")
    ap.add_argument("--config", default="config.paper.yaml")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in cfg["models"]}
    judges = []
    for name in [n.strip() for n in args.judges.split(",") if n.strip()]:
        if name not in by_name:
            sys.exit(f"judge '{name}' is not in {args.config}")
        entry = by_name[name]
        judges.append(build_client(entry["name"], entry["provider"], entry["model_id"]))

    src = ROOT / args.source
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((src / "raw_results.json").read_text(encoding="utf-8"))

    out_path = out / "raw_results.json"
    done = {}
    if out_path.exists():
        for r in json.loads(out_path.read_text(encoding="utf-8")):
            done[(r["model"], r["task"], r["item_id"])] = r

    tasks: dict = {}
    results, skipped, judged = [], 0, 0
    for record in raw:
        key = (record["model"], record["task"], record["item_id"])
        if key in done:
            results.append(done[key])
            continue
        new = copy.deepcopy(record)
        empty = any(not (t or "").strip() for t in record["outputs"].values())
        task_obj = tasks.setdefault(record["task"], build_task(record["task"]))
        if record.get("error") or empty or not task_obj.needs_judge:
            # The correction: an output with no response is never judged. It is
            # still reported, and the audit counts it, so excluding it here
            # removes a distorted score rather than removing the item.
            new["judge_records"] = []
            new["judging_skipped"] = True
            new["unparsed_judges"] = []
            new["avg_judge_score"] = None
            skipped += 1
        else:
            task = task_obj
            item = next(it for it in task.get_items() if it.id == record["item_id"])
            verdicts = []
            for judge in judges:
                res = judge_item(judge, task, item, record["outputs"])
                verdicts.append({
                    "judge_name": judge.name, "score": res.score,
                    "reason": (res.reason or "")[:300],
                    "call_failed": not res.response.ok,
                })
            new["judge_records"] = verdicts
            new["judging_skipped"] = False
            new["unparsed_judges"] = [v["judge_name"] for v in verdicts if v["score"] is None]
            scored = [v["score"] for v in verdicts if v["score"] is not None]
            new["avg_judge_score"] = (sum(scored) / len(scored)) if scored else None
            judged += 1
            if judged % 5 == 0:
                print(f"  judged {judged} ... {key[0]}/{key[2]}", flush=True)
        results.append(new)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for extra in ("human_ratings.csv", "human_ratings_aditya.csv", "human_ratings_bob.csv"):
        if (src / extra).exists():
            (out / extra).write_bytes((src / extra).read_bytes())

    print(f"\n{len(results)} records -> {out_path}")
    print(f"  judged under the corrected pipeline: {judged}")
    print(f"  not judged (empty response or API error): {skipped}")
    print(f"  judges: {', '.join(j.name for j in judges)}")


if __name__ == "__main__":
    main()
