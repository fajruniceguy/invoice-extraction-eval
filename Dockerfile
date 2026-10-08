# One image for both compose services: `api` (uvicorn) and `eval` (scripts/eval_api.py).
# Runtime dependencies only (no dev, degrade or tools groups). Nothing secret is copied in: the
# build context is an allow-list (.dockerignore) and the API key arrives at run time through the
# environment (compose `env_file: .env`).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /app

# dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-default-groups --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY data/manifest.csv ./data/manifest.csv
COPY data/pages ./data/pages
COPY ground_truth ./ground_truth
RUN uv sync --frozen --no-dev --no-default-groups

# the response cache, the call log and the eval runs are written here (compose mounts ./outputs)
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/outputs \
    && chown -R app:app /app/outputs
USER app

EXPOSE 8000
CMD ["uvicorn", "--factory", "docextract.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
