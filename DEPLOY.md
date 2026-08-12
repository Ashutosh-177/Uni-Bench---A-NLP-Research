# Deploying UniBench-NLP to the public internet

This covers taking the dashboard from your machine to `https://uni-bench.com`
(or any domain). Read **"Before you deploy"** first — it's not optional.

## Before you deploy

This app has no access control beyond what `webapp/backend/main.py`'s
`BasicAuthMiddleware` provides, and it holds real, spend-capable API keys.
Two things are non-negotiable before the URL is public:

1. **Set `APP_PASSWORD`** (and optionally change `APP_USERNAME` from the
   default `admin`) as an environment variable on whatever host you use.
   Without it, `/api/run` is reachable by anyone and will spend your Groq/
   Gemini quota on their behalf. `run_web.py` prints a loud warning at
   startup if it detects a cloud environment with no password set — don't
   ignore that warning.
2. **Only ever access the deployed URL over `https://`.** HTTP Basic Auth
   sends credentials base64-*encoded*, not encrypted — fine over TLS,
   trivially readable over plain HTTP. Render/Railway/Fly all terminate
   TLS for you automatically on their default domains and on custom
   domains once DNS is verified; just don't disable it.

## 1. Get the code onto GitHub

Render (and Railway, and Fly) deploy from a git repository.

```bash
cd "UniBench-NLP"
git init
git add .
git commit -m "Initial commit"
```

Then create an empty repository on github.com (New repository, no README/
.gitignore -- you already have one), and:

```bash
git remote add origin https://github.com/<your-username>/unibench-nlp.git
git branch -M main
git push -u origin main
```

`.env` is already gitignored — double check `git status` doesn't show it
before you push. Your API keys should never reach GitHub.

## 2. Deploy on Render

**Option A — Blueprint (uses `render.yaml`, least clicking):**
1. [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect the GitHub repo you just pushed.
3. Render reads `render.yaml` and proposes the service. Confirm.
4. It will prompt you for the env vars marked `sync: false`: `GROQ_API_KEY`,
   `GEMINI_API_KEY`, `APP_USERNAME`, `APP_PASSWORD`. Fill all four in —
   **`APP_PASSWORD` is not optional here.**
5. Deploy. First build takes a few minutes (installing pandas/scipy/
   matplotlib/etc.).

**Option B — manual web service (no `render.yaml`):**
1. **New** → **Web Service** → connect the repo.
2. Environment: **Docker** (it'll detect the `Dockerfile` automatically).
3. Instance type: Free (or Starter, ~$7/mo, for always-on — the free tier
   spins down after 15 min idle and takes ~30s to wake back up on the
   next request).
4. Add the same four environment variables as above under **Environment**.
5. Create Web Service.

Either way, you'll get a working `https://unibench-nlp.onrender.com`-style
URL before touching the domain at all — visit it and confirm the Basic
Auth prompt appears and your password works before moving on.

## 3. Point uni-bench.com at it

1. In Render: the service's **Settings → Custom Domains** → **Add
   Custom Domain** → enter `uni-bench.com` (and `www.uni-bench.com` if you
   want both). Render shows you the exact DNS record to create (typically
   a `CNAME` for `www` and an `A` record, or Render's `ANAME`/`ALIAS`
   equivalent, for the bare apex domain).
2. In IFreeDomains: open the **DNS Management** panel for `uni-bench.com`
   (the add-on you already have in your cart) and add the record Render
   gave you.
3. DNS propagation is usually minutes, occasionally up to ~24h. Render
   auto-issues an HTTPS certificate for the domain once it verifies.

## 4. Persistent storage — read this before relying on results

Render's free (and Starter) plans use an **ephemeral filesystem**: every
redeploy (including ones you didn't trigger, like a routine restart)
wipes anything the app wrote to disk. For this app, that means
`config.yaml` edits made through the dashboard and everything in
`results/` (raw results, calibration ratings, the leaderboard) **do not
survive a redeploy** unless you do one of:

- **Accept it.** Fine if this deployment is a live demo/showcase and your
  "real" benchmark runs happen locally, with the CLI, where results are
  permanent.
- **Add a persistent disk** (uncomment the `disk:` block in `render.yaml`,
  or add one in the Render dashboard under the service's **Disks** tab —
  requires a paid plan). Mount it, then point `results_dir` in
  `config.yaml` and the `CONFIG_PATH` the app writes to at the mounted
  path, so both survive redeploys.

## 5. Updating the deployed app later

```bash
git add -A
git commit -m "describe the change"
git push
```

Render auto-redeploys on every push to the branch you connected. No
further steps needed on the Render side.

## Cost summary

| Item | Cost |
|---|---|
| Domain (`uni-bench.com`, IFreeDomains) | Whatever they charge after any free first year — confirm the renewal price |
| Render free tier | $0, but sleeps after 15 min idle (~30s cold start) |
| Render Starter (always-on) | ~$7/mo |
| Render persistent disk | Additional, scales with size (only needed if you want results to survive redeploys) |
| Groq / Gemini API usage | Free tier as configured; real cost only if you exceed it |
