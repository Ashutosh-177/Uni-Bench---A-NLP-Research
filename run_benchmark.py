#!/usr/bin/env python
"""UniBench-NLP command-line entrypoint.

Usage:
    python run_benchmark.py run       [--config config.yaml]
    python run_benchmark.py calibrate [--config config.yaml] [--n N]
    python run_benchmark.py report    [--config config.yaml]

Typical workflow: run these three in order. `run` calls every configured
model on every task and saves raw_results.json. `calibrate` has YOU rate
the outputs blind (without seeing the model or any judge score) so AI-judge
bias can be corrected. `report` produces the final leaderboard, Pareto
summary, data-completeness audit, and chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Windows terminals often default to the cp1252 codepage, which cannot
# encode some characters report.py may emit; force UTF-8 on stdout/stderr
# so console output never crashes the run regardless of terminal settings.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv

from unibench.calibrate import run_calibration
from unibench.models import build_client
from unibench.report import generate_report
from unibench.runner import run_benchmark
from unibench.tasks import build_task


def load_config(config_path: str):
    load_dotenv()  # populate os.environ from .env, if present
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    clients = {
        m["name"]: build_client(m["name"], m["provider"], m["model_id"])
        for m in cfg["models"]
    }
    judges = [clients[name] for name in cfg.get("judges", [])]
    tasks = [build_task(name) for name in cfg.get("tasks", [])]
    results_dir = Path(cfg.get("results_dir", "results"))
    return cfg, clients, judges, tasks, results_dir


def cmd_run(args: argparse.Namespace) -> None:
    cfg, clients, judges, tasks, results_dir = load_config(args.config)
    run_cfg = cfg.get("run", {})
    print(f"Models: {list(clients)}")
    print(f"Judges: {[j.name for j in judges]}")
    print(f"Tasks:  {[t.name for t in tasks]}\n")
    run_benchmark(
        models=list(clients.values()), tasks=tasks, judges=judges,
        results_dir=results_dir,
        temperature=run_cfg.get("temperature", 0.0),
        max_tokens=run_cfg.get("max_tokens", 400),
        resume=getattr(args, "resume", False),
    )
    print(f"\nDone. Raw results in {results_dir/'raw_results.json'}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    _, _, _, _, results_dir = load_config(args.config)
    run_calibration(results_dir, limit=args.n, rater=args.rater)


def cmd_report(args: argparse.Namespace) -> None:
    _, _, _, _, results_dir = load_config(args.config)
    generate_report(results_dir)


def main() -> None:
    # `--config` lives on this shared parent parser so it works BOTH before
    # and after the subcommand (`run_benchmark.py --config x.yaml run` and
    # `run_benchmark.py run --config x.yaml` both work) -- argparse does not
    # propagate an argument defined only on the top-level parser to args
    # typed after the subcommand, which is the fix this addresses.
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")

    parser = argparse.ArgumentParser(description="UniBench-NLP: cross-domain, bias-corrected LLM benchmarking.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", parents=[config_parent],
        help="Run every configured model on every configured task.")
    run_parser.add_argument(
        "--resume", action="store_true",
        help="Keep the records already in raw_results.json and only run what is "
             "missing. Use after a run is interrupted -- without it the file is "
             "rebuilt from scratch and every completed call is paid for twice.")

    calibrate_parser = subparsers.add_parser("calibrate", parents=[config_parent],
                                              help="Blind human rating of the outputs, for judge bias correction.")
    calibrate_parser.add_argument("--n", type=int, default=None,
                                  help="Rate at most N outputs this session (default: all not yet rated).")
    calibrate_parser.add_argument("--rater", default=None,
                                  help="Name of an additional rater; saves to a separate file used only "
                                       "for inter-rater agreement, not for judge calibration.")

    subparsers.add_parser("report", parents=[config_parent],
                           help="Build the leaderboard, Pareto summary, and chart.")

    args = parser.parse_args()
    {"run": cmd_run, "calibrate": cmd_calibrate, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
