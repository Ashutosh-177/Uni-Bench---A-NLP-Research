# UniBench-NLP web dashboard -- portable container image (Render, Railway,
# Fly.io, or plain `docker run` all work from this same image).
#
# Build from THIS directory (UniBench-NLP/), not the parent folder:
#   docker build -t unibench-nlp .
#   docker run -p 8000:8000 --env-file .env unibench-nlp
#
# Secrets (GROQ_API_KEY, GEMINI_API_KEY, APP_USERNAME, APP_PASSWORD) are
# NEVER baked into the image -- .dockerignore excludes .env, and every
# hosting platform's docs (see DEPLOY.md) inject them as runtime
# environment variables instead.

FROM python:3.12-slim

WORKDIR /app

# Installed separately from the rest of the source so Docker's layer cache
# skips reinstalling ~15 packages (fastapi, pandas, scipy, matplotlib, ...)
# on every code change -- only reruns when requirements*.txt actually change.
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . .

# Render/Railway/Fly inject their own $PORT at runtime and override this;
# it's set here only so a bare `docker run -p 8000:8000 ...` (testing the
# image locally, no platform involved) works out of the box. Presence of
# $PORT is also what tells run_web.py to bind 0.0.0.0 instead of
# localhost-only -- see run_web.py's `_ON_CLOUD_HOST` check.
ENV PORT=8000
EXPOSE 8000

CMD ["python", "run_web.py"]
