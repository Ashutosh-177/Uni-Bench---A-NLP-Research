#!/usr/bin/env python
"""Starts the UniBench-NLP web dashboard and opens it in your browser.

Usage:
    python run_web.py [--port 8000] [--host 127.0.0.1] [--no-browser]

Cloud hosts (Render/Railway/Fly) set a $PORT env var and require binding
to 0.0.0.0 so their reverse proxy can reach the process -- when $PORT is
present, both defaults switch automatically, so the SAME command
(`python run_web.py`) works unmodified locally and in the Dockerfile/
render.yaml. Explicit --host/--port flags always win over both.
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser

import uvicorn

_ON_CLOUD_HOST = "PORT" in os.environ  # set by Render/Railway/Fly, not by local dev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", default="0.0.0.0" if _ON_CLOUD_HOST else "127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", default=_ON_CLOUD_HOST)
    args = parser.parse_args()

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    if _ON_CLOUD_HOST and not os.environ.get("APP_PASSWORD"):
        print("\n*** WARNING: APP_PASSWORD is not set. This dashboard is being served "
              "on a cloud host with NO authentication -- anyone who finds the URL can "
              "trigger paid API calls on your keys. Set APP_PASSWORD before this is "
              "publicly reachable. See DEPLOY.md. ***\n")

    print(f"\nUniBench-NLP dashboard starting on {args.host}:{args.port}\n")
    uvicorn.run("webapp.backend.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
