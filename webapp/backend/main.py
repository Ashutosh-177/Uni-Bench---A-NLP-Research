"""FastAPI backend for the UniBench-NLP web dashboard.

Pure HTTP plumbing over the existing `unibench` package -- no benchmarking
logic lives here. Config CRUD, triggering a real run in a background
thread, polling progress, serving the same report data `report.py`'s CLI
computes, and calibration sampling/submission. Serves the static frontend
from ../frontend at "/".

Run via `python run_web.py` from the project root (handles sys.path,
.env loading, and opening the browser); this module assumes that's already
been done if imported directly by uvicorn.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from unibench.calibrate import append_calibration_row, sample_calibration_candidates  # noqa: E402
from unibench.models import PROVIDER_REGISTRY, build_client  # noqa: E402
from unibench.report import compute_report_data  # noqa: E402
from unibench.runner import run_benchmark  # noqa: E402
from unibench.tasks import TASK_REGISTRY, build_task  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ============================================================
# Password gate
# ============================================================
# This app has NO other access control: `/api/run` spends real money on
# whichever API keys are in the environment, and `/api/config` can
# overwrite config.yaml outright. That's an acceptable risk on localhost,
# but NOT once this is reachable from the public internet (e.g. deployed
# behind a domain) -- so if APP_PASSWORD is set, every request (except the
# health check and CORS preflight) must present it via HTTP Basic Auth.
#
# If APP_PASSWORD is unset, auth is skipped entirely -- this keeps local
# `python run_web.py` frictionless. Deployment docs (DEPLOY.md) make
# setting APP_PASSWORD a required step, not optional, before going live.
#
# HTTP Basic sends credentials base64-ENCODED, not encrypted -- this is
# only safe over HTTPS. Every recommended host (Render/Railway/Fly)
# terminates TLS for you automatically; never disable that in production.
APP_USERNAME = os.environ.get("APP_USERNAME", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD")  # None => auth disabled
_UNAUTHED_PATHS = {"/api/health"}


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not APP_PASSWORD or request.method == "OPTIONS" or request.url.path in _UNAUTHED_PATHS:
            return await call_next(request)

        header = request.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:]).decode("utf-8")
                username, _, password = decoded.partition(":")
            except (binascii.Error, UnicodeDecodeError):
                username, password = "", ""
            # constant-time comparisons -- avoid leaking match-length via timing
            if secrets.compare_digest(username, APP_USERNAME) and secrets.compare_digest(password, APP_PASSWORD):
                return await call_next(request)

        return Response(
            status_code=401,
            content="Authentication required.",
            headers={"WWW-Authenticate": 'Basic realm="UniBench-NLP"'},
        )


app = FastAPI(title="UniBench-NLP")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(BasicAuthMiddleware)  # registered last -> outermost -> runs first


# ============================================================
# In-memory run state -- single-user local app, no DB needed.
# A real run is a blocking, minutes-long loop of API calls, so it executes
# in a background thread; the frontend polls /api/run/progress for updates
# (simple, robust, and plenty smooth at a ~500ms poll cadence for a local
# single-user dashboard -- no websocket/SSE plumbing needed).
# ============================================================

class RunState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.events: List[dict] = []
        self.total = 0
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def reset(self, total: int) -> None:
        with self.lock:
            self.running = True
            self.events = []
            self.total = total
            self.error = None
            self.started_at = time.time()
            self.finished_at = None

    def push(self, event: dict) -> None:
        with self.lock:
            self.events.append(event)

    def finish(self, error: Optional[str] = None) -> None:
        with self.lock:
            self.running = False
            self.error = error
            self.finished_at = time.time()

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            return {
                "running": self.running, "total": self.total,
                "completed": len(self.events), "error": self.error,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "events": self.events[since:],
            }


run_state = RunState()


# ============================================================
# Config helpers
# ============================================================

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise HTTPException(404, f"{CONFIG_PATH} not found")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


class ModelEntry(BaseModel):
    name: str
    provider: str
    model_id: str


class ConfigUpdate(BaseModel):
    models: List[ModelEntry]
    judges: List[str]
    tasks: List[str]
    run: dict
    calibration: dict = {"sample_size": 10}
    results_dir: str = "results"


# ============================================================
# Basic endpoints
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return _load_config()


@app.put("/api/config")
def put_config(cfg: ConfigUpdate):
    _save_config(cfg.model_dump())
    return {"ok": True}


@app.get("/api/registry")
def get_registry():
    """What's actually pluggable right now -- lets the UI build its model/
    task pickers from ground truth instead of a hardcoded list that could
    drift from the codebase."""
    return {
        "providers": list(PROVIDER_REGISTRY.keys()),
        "tasks": list(TASK_REGISTRY.keys()),
    }


@app.get("/api/groq/models")
def groq_models():
    """Live model list from Groq's own /models endpoint -- this is exactly
    the check that caught the stale `qwen/qwen3-32b` id during development;
    the UI uses this so a user never has to guess a model id by hand."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise HTTPException(400, "GROQ_API_KEY not set in .env")
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"}, timeout=15,
    )
    if resp.status_code != 200:
        raise HTTPException(502, f"Groq error {resp.status_code}: {resp.text[:200]}")
    return sorted(m["id"] for m in resp.json().get("data", []))


# ============================================================
# Run
# ============================================================

def _do_run(models_cfg: List[dict], judge_names: List[str], task_names: List[str],
            run_cfg: dict, results_dir: str) -> None:
    try:
        clients = {m["name"]: build_client(m["name"], m["provider"], m["model_id"]) for m in models_cfg}
        judges = [clients[n] for n in judge_names if n in clients]
        tasks = [build_task(n) for n in task_names]
        total = sum(len(t.get_items()) for t in tasks) * len(clients)
        run_state.reset(total)
        run_benchmark(
            models=list(clients.values()), tasks=tasks, judges=judges,
            results_dir=Path(results_dir),
            temperature=run_cfg.get("temperature", 0.0),
            max_tokens=run_cfg.get("max_tokens", 800),
            verbose=False,
            on_progress=run_state.push,
        )
        run_state.finish()
    except Exception as exc:  # noqa: BLE001 -- surface ANY failure to the UI, don't crash the thread silently
        run_state.finish(error=str(exc))


@app.post("/api/run")
def start_run():
    if run_state.running:
        raise HTTPException(409, "A run is already in progress.")
    cfg = _load_config()
    if not cfg.get("models"):
        raise HTTPException(400, "No models configured.")
    thread = threading.Thread(
        target=_do_run,
        args=(cfg["models"], cfg.get("judges", []), cfg.get("tasks", []),
              cfg.get("run", {}), cfg.get("results_dir", "results")),
        daemon=True,
    )
    thread.start()
    return {"started": True}


@app.get("/api/run/progress")
def run_progress(since: int = 0):
    return run_state.snapshot(since)


# ============================================================
# Results
# ============================================================

@app.get("/api/results/summary")
def results_summary():
    cfg = _load_config()
    results_dir = Path(cfg.get("results_dir", "results"))
    try:
        return compute_report_data(results_dir)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/results/raw")
def results_raw():
    cfg = _load_config()
    results_dir = Path(cfg.get("results_dir", "results"))
    path = results_dir / "raw_results.json"
    if not path.exists():
        raise HTTPException(404, "No results yet -- run a benchmark first.")
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# Calibration
# ============================================================

@app.get("/api/calibration/sample")
def calibration_sample(n: int = 8):
    cfg = _load_config()
    results_dir = Path(cfg.get("results_dir", "results"))
    try:
        return sample_calibration_candidates(results_dir, n=n, seed=None)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


class CalibrationSubmit(BaseModel):
    model: str
    task: str
    item_id: str
    judge_name: str
    ai_score: float
    ai_reason: str
    human_score: float


@app.post("/api/calibration/submit")
def calibration_submit(row: CalibrationSubmit):
    cfg = _load_config()
    results_dir = Path(cfg.get("results_dir", "results"))
    append_calibration_row(results_dir, row.model_dump())
    return {"ok": True}


# ============================================================
# Static frontend (mounted LAST -- routes above take priority)
# ============================================================

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
