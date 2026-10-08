"""Mutants for the Docker setup (Dockerfile, .dockerignore, docker-compose.yml).

  python -m uv run python tools/mutation_check.py tests/mutants/docker_files.py --copy LIST

LIST is the tool's default copy plus the files these tests read (the default has no Docker files
and no uv.lock): src,tools,tests,scripts,docs,ground_truth,.claude,pyproject.toml,.gitignore,
uv.lock,.python-version,Dockerfile,.dockerignore,docker-compose.yml (one argument, no spaces).

Every entry breaks one rule on purpose; tests/test_docker_files.py must notice.
This file is not collected by pytest (its name does not start with `test_`).
"""

D = "Dockerfile"
I = ".dockerignore"  # noqa: E741
C = "docker-compose.yml"
T = ["tests/test_docker_files.py"]

MUTANTS = [
    (
        D,
        T,
        "the base is not Python 3.11",
        "FROM python:3.11-slim-bookworm",
        "FROM python:3.12-slim",
    ),
    (
        D,
        T,
        "the lock file is not enforced",
        "uv sync --frozen --no-dev --no-default-groups --no-install-project",
        "uv sync --no-dev --no-default-groups --no-install-project",
    ),
    (
        D,
        T,
        "dev groups are installed",
        "uv sync --frozen --no-dev --no-default-groups\n",
        "uv sync --frozen --no-default-groups\n",
    ),
    (D, T, "the image runs as root", "USER app", "USER root"),
    (D, T, "the whole context is copied", "COPY scripts ./scripts", "COPY . ."),
    (
        D,
        T,
        "the label drafts are copied",
        "COPY ground_truth ./ground_truth",
        "COPY ground_truth ./ground_truth\nCOPY ground_truth/drafts ./drafts",
    ),
    (D, T, "the server listens on localhost only", '"--host", "0.0.0.0"', '"--host", "127.0.0.1"'),
    (
        D,
        T,
        "a key is baked into the image",
        "ENV PYTHONUNBUFFERED=1",
        "ENV LLM_API_KEY=abc\nENV PYTHONUNBUFFERED=1",
    ),
    (I, T, "the ignore file is a deny-list", "*\n\n!pyproject.toml", "!pyproject.toml"),
    (I, T, "the private data is let back in", "!ground_truth", "!ground_truth\n!data/private"),
    (I, T, "the drafts are let in", "ground_truth/drafts", "ground_truth/old-drafts"),
    (I, T, "the pages are not allowed", "!data/pages", "!data/page"),
    (C, T, "the api has no env file", "    env_file: .env\n", ""),
    (C, T, "the api has no healthcheck", "    healthcheck:", "    xhealthcheck:"),
    (
        C,
        T,
        "the eval does not wait for health",
        "condition: service_healthy",
        "condition: service_started",
    ),
    (
        C,
        T,
        "the eval gets the key",
        "    environment:\n      API_URL: http://api:8000",
        "    env_file: .env\n    environment:\n      API_URL: http://api:8000",
    ),
    (C, T, "the eval starts with a plain up", 'profiles: ["eval"]', "# no profile"),
    (
        C,
        T,
        "the eval gets a command that run replaces",
        "    entrypoint:",
        '    command: ["--split", "test"]\n    entrypoint:',
    ),
    (
        C,
        T,
        "outputs are not mounted for the eval",
        "    volumes:\n      - ./outputs:/app/outputs\n    entrypoint",
        "    entrypoint",
    ),
]
