#!/usr/bin/env python
"""Quick connectivity smoke test -- one tiny call per backend configured in
config.yaml, so you can confirm your API keys actually work BEFORE burning
a full `run_benchmark.py run` on the whole task suite.

Usage:
    python scripts/check_connection.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from dotenv import load_dotenv

from unibench.models import build_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    print(f"Testing {len(cfg['models'])} model(s) from {args.config}...\n")
    all_ok = True
    for m in cfg["models"]:
        client = build_client(m["name"], m["provider"], m["model_id"])
        response = client.generate("Reply with exactly the two words: connection ok",
                                    temperature=0.0, max_tokens=20)
        if response.ok:
            print(f"[OK]   {m['name']:20s} ({m['provider']}/{m['model_id']}) -> {response.text.strip()!r} "
                  f"({response.latency_seconds:.2f}s, {response.total_tokens} tokens)")
        else:
            all_ok = False
            print(f"[FAIL] {m['name']:20s} ({m['provider']}/{m['model_id']}) -> {response.error}")

    print("\nAll backends reachable." if all_ok else "\nSome backends failed -- see [FAIL] lines above.")


if __name__ == "__main__":
    main()
