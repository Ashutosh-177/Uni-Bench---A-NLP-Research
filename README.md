# UniBench-NLP

A cross-domain, bias-corrected, efficiency-aware benchmarking framework for
comparing NLP-capable AI models — the implementation of the "Proposed
Approach" in `literature_review.tex`, addressing the research gap
identified there: existing AI-model comparisons are locked inside single
application domains, use uncorrected AI-judge scores, ignore multi-step
failure and low-resource performance, and treat efficiency as a footnote.

This runs the **same** pool of models through the **same** tasks under the
**same** prompting rules, corrects AI-judge bias against a small human-rated
subset (the way PSE-Bench does, but generalized to any judge), and reports
results as a **Pareto ranking** across accuracy, fairness, and cost —
instead of a single misleading leaderboard number.

## What it does

1. **`run`** — calls every configured model (proprietary API + open-weight)
   on every configured task (fairness probe, summarization, …), scores
   responses with automatic metrics *and* a panel of AI judges, and logs
   every call's tokens/latency/cost. Saves `results/raw_results.json` and
   `results/run_log.csv` incrementally, so a crash mid-run loses nothing.
2. **`calibrate`** — shows *you* a random sample of AI-judge-scored
   responses and asks for your own 0–10 score. Saved to
   `results/human_calibration.csv`.
3. **`report`** — uses your calibration scores to correct each judge's bias
   (regression if you gave ≥5 ratings for a judge, mean-offset otherwise,
   explicitly marked "uncalibrated" if you gave none), computes the
   Pareto-optimal model(s) per task, runs a Friedman significance test, and
   writes `results/leaderboard.csv`, `results/report.md`,
   `results/report.html`, and `results/accuracy_vs_cost.png`.

## Setup

```bash
cd UniBench-NLP
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste in your keys
```

Get free keys:
- Groq: https://console.groq.com/keys
- Gemini: https://aistudio.google.com/apikey (see "Known issues" below --
  this free tier is not available in every country, regardless of how the
  key is created)

No keys yet? Set a model's `provider: mock` in `config.yaml` to smoke-test
the whole pipeline with a fake, zero-cost model first.

## Run it

```bash
python run_benchmark.py run
python run_benchmark.py calibrate --n 10
python run_benchmark.py report
```

Open `results/report.html` for the final dashboard.

## Project layout

```
config.yaml              # model pool, judges, tasks, run settings
.env.example              # copy to .env, add your API keys
run_benchmark.py          # CLI entrypoint (run / calibrate / report)
data/                     # curated MVP datasets (fairness pairs, summaries)
unibench/
  models/                 # ModelClient interface + Groq/Gemini/Mock backends
  tasks/                  # Task interface + fairness & summarization tasks
  evaluation/
    ai_judge.py            # multi-AI-judge scoring
    bias_calibration.py    # judge bias correction vs. human subset
    aggregator.py           # Pareto ranking + Friedman significance test
  telemetry.py             # incremental cost/latency/token logging
  runner.py                # orchestrates models x tasks x judges
  report.py                # builds leaderboard, chart, and HTML report
results/                  # generated output (gitignored)
```

## Extending it (matches the paper's full-scope design)

The current scope is deliberately small (3 models, 2 tasks, hand-authored
data) so it's cheap and fast to run end-to-end. Each of these extensions
only requires adding a new file — the rest of the framework doesn't change,
because everything talks to the `ModelClient` and `Task` interfaces:

- **New model backend** (OpenAI, Anthropic, local Ollama): add a file in
  `unibench/models/`, register it in `PROVIDER_REGISTRY`
  (`unibench/models/__init__.py`).
- **New task** (NER, QA, low-resource/code-mixed translation, a multi-step
  agentic task): add a file in `unibench/tasks/`, register it in
  `TASK_REGISTRY` (`unibench/tasks/__init__.py`). Swap the MVP
  hand-authored summarization set for a real CNN/DailyMail or XSum subset
  via the HuggingFace `datasets` library the same way.
- **More fairness axes**: add entries to `data/fairness_pairs.json`.

## Known issues

- **Gemini keys can 429 with `limit: 0` -- try a fresh key before assuming
  it's your region.** Two of three keys generated during development hit
  HTTP 429 with `limit: 0` on every quota metric (distinct from a normal
  "you used up your quota" 429), which usually means the free tier isn't
  enabled for that Google account/project. A third key, generated fresh via
  aistudio.google.com/apikey, worked immediately with no billing changes --
  so this is worth retrying with a new key/account before concluding it's a
  hard regional block. If it persists across multiple fresh keys, enabling
  billing on the key's GCP project (console.cloud.google.com -> Billing) is
  the fallback.
- **Gemini model ids go stale fast.** `gemini-2.0-flash` AND
  `gemini-2.5-flash` were both already retired ("no longer available to new
  users") by the time this was tested -- `config.yaml` currently pins
  `gemini-3.6-flash`, confirmed against the live model list, not assumed.
  If you get a 404, requery
  `https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY`
  and pick a current non-"-preview" `-flash` entry.
- **Reasoning models need a larger `max_tokens` budget than you'd think.**
  Qwen's thinking mode was observed spending its *entire* completion budget
  on hidden `<think>...</think>` reasoning at `max_tokens=400`, producing
  empty output on 100% of summarization items -- not a rare edge case, every
  single one. `config.yaml` defaults to 800 for this reason; if you add a
  new reasoning-heavy model, re-verify this is still enough headroom.
- **Groq's free-tier rate limit (8000 tokens/min per model) is real and
  will be hit** on a full run across multiple tasks -- `unibench/models/base.py`
  retries automatically with backoff, so this should be transparent, but if
  you add many more models/tasks you may want to raise
  `max_rate_limit_retries` or add a fixed delay between calls.

## Honesty notes (please read before citing results)

- The fairness-task metrics (`response_divergence`, `length_asymmetry`,
  `tone_gap`) are a transparent, simple proxy — TF-IDF similarity and a
  small hand-built word list — **not** a validated, peer-reviewed bias
  metric. They're directional signals, not certified scores. See the
  docstring in `unibench/tasks/fairness_task.py`.
- Cost is estimated (`$0.00` for free-tier Groq/Gemini usage in the default
  config) and is **not** billing-accurate.
- The Friedman significance test explicitly flags low statistical power
  when there are few items — with the MVP-scale sample sizes here, treat
  any "significant" result as directional, not conclusive; the full-scope
  design in the paper calls for larger task sets before drawing firm
  conclusions.
