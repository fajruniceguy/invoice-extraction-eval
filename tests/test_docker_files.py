"""The Docker setup, checked as text (no Docker daemon needed): what the image contains, what it
must never contain, how the two services are wired, and that the runtime-only dependency set the
image installs is enough to import and score."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from ingest_support import ROOT

DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8")
COMPOSE: dict[str, Any] = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def instructions(name: str) -> list[str]:
    """The arguments of every `name` instruction (continuation lines joined)."""
    text = re.sub(r"\\\r?\n", " ", DOCKERFILE)
    return [
        line.split(None, 1)[1].strip()
        for line in text.splitlines()
        if line.upper().startswith(name.upper() + " ")
    ]


def ignore_rules() -> tuple[list[str], list[str]]:
    """(excluded patterns, re-included patterns) of .dockerignore, comments dropped."""
    rules = [line.strip() for line in DOCKERIGNORE.splitlines() if line.strip()]
    rules = [rule for rule in rules if not rule.startswith("#")]
    return [r for r in rules if not r.startswith("!")], [r[1:] for r in rules if r.startswith("!")]


class TestImage:
    def test_a_pinned_python_3_11_base(self) -> None:
        (base,) = instructions("FROM")
        assert re.fullmatch(r"python:3\.11[\w.-]*", base), base

    def test_dependencies_come_from_the_lock_file_without_dev_groups(self) -> None:
        syncs = [line for line in instructions("RUN") if "uv sync" in line]
        assert len(syncs) == 2  # the dependencies first (cache layer), then the project
        for line in syncs:
            assert "--frozen" in line and "--no-dev" in line and "--no-default-groups" in line
        assert "--no-install-project" in syncs[0]
        assert re.search(r"uv==\d+\.\d+\.\d+", DOCKERFILE)  # uv itself is pinned

    def test_it_runs_as_a_non_root_user(self) -> None:
        users = instructions("USER")
        assert users and users[-1] not in ("root", "0")

    def test_it_serves_the_api_factory_on_all_interfaces(self) -> None:
        (command,) = instructions("CMD")
        assert "docextract.api.app:create_app" in command and "--factory" in command
        assert "0.0.0.0" in command

    def test_it_copies_no_secret_and_no_private_data(self) -> None:
        joined = " ".join(instructions("COPY") + instructions("ADD"))
        for forbidden in (".env", "private", "outputs", ".git", "drafts"):
            assert forbidden not in joined, forbidden
        assert instructions("COPY")  # something is copied
        assert "COPY . " not in DOCKERFILE and "COPY ./ " not in DOCKERFILE
        assert not re.search(r"^\s*(ENV|ARG)\s+\S*(KEY|TOKEN|SECRET|PASSWORD)", DOCKERFILE, re.M)

    def test_every_copied_source_is_allowed_by_dockerignore(self) -> None:
        excluded, allowed = ignore_rules()
        assert excluded[0] == "*"  # an allow-list: everything is out unless named
        for line in instructions("COPY"):
            *sources, _ = [part for part in line.split() if not part.startswith("--")]
            for source in sources:
                assert any(
                    source.removeprefix("./") == rule
                    or source.removeprefix("./").startswith(rule + "/")
                    for rule in allowed
                ), f"{source} is copied but .dockerignore does not let it in"


class TestDockerignore:
    def test_nothing_secret_or_private_is_ever_let_back_in(self) -> None:
        _, allowed = ignore_rules()
        for rule in allowed:
            for forbidden in (".env", "private", "outputs", ".git", ".venv", "drafts"):
                assert forbidden not in rule, f"!{rule} re-includes {forbidden}"
        assert not any(rule.startswith("data") and rule.rstrip("/") == "data" for rule in allowed)

    def test_the_label_drafts_stay_out(self) -> None:
        excluded, _ = ignore_rules()
        assert "ground_truth/drafts" in excluded


class TestCompose:
    def test_the_api_service_has_the_env_file_a_healthcheck_and_the_outputs_mount(self) -> None:
        api = COMPOSE["services"]["api"]
        assert api["env_file"] == ".env"
        assert "./outputs:/app/outputs" in api["volumes"]
        assert "8000:8000" in api["ports"]
        check = api["healthcheck"]["test"]
        assert "/health" in " ".join(check)
        assert api["healthcheck"].get("retries", 0) >= 3

    def test_the_eval_service_waits_for_a_healthy_api_and_never_gets_the_model_key(self) -> None:
        evaluation = COMPOSE["services"]["eval"]
        assert evaluation["depends_on"]["api"]["condition"] == "service_healthy"
        assert "env_file" not in evaluation
        assert evaluation["environment"]["API_URL"] == "http://api:8000"
        text = yaml.safe_dump(evaluation)
        for name in ("LLM_API_KEY", "PRELABEL"):
            assert name not in text
        assert evaluation["entrypoint"][-1] == "scripts/eval_api.py"
        # no `command`: arguments given to `run` replace it, so the default lives in the script
        assert "command" not in evaluation
        assert "./outputs:/app/outputs" in evaluation["volumes"]

    def test_plain_up_does_not_start_the_eval_and_both_use_one_image(self) -> None:
        services = COMPOSE["services"]
        assert services["eval"]["profiles"] == ["eval"]
        assert "profiles" not in services["api"]
        assert services["api"]["image"] == services["eval"]["image"]


@pytest.mark.slow
def test_the_runtime_only_dependency_set_is_enough_to_import_and_score(tmp_path: Path) -> None:
    """`uv sync --frozen --no-dev --no-default-groups` (what the image does) into a fresh venv:
    the API, the eval client and the scorer import there, and numpy/OpenCV (the degrade group) and
    streamlit (tools) are not needed."""
    environment = tmp_path / "venv"
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(environment),
        "UV_LINK_MODE": "copy",
        "UV_PYTHON_DOWNLOADS": "never",
    }
    uv = os.environ.get("UV") or shutil.which(
        "uv"
    )  # `uv run` sets UV; python -m uv is not in a venv
    if not uv:
        pytest.skip("uv is not on PATH and not run through `uv run`")
    sync = subprocess.run(
        [uv, "sync", "--frozen", "--no-dev", "--no-default-groups", "--offline"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if sync.returncode != 0 and "offline" in (sync.stderr or "").lower():
        pytest.skip("the wheels are not in uv's cache and the test runs offline")
    assert sync.returncode == 0, sync.stderr
    python = environment / ("Scripts" if os.name == "nt" else "bin") / "python"
    code = (
        "import docextract.api.app, evalkit.scorecard, evalkit.compare, sys\n"
        "for name in ('numpy', 'cv2', 'streamlit', 'pytest'):\n"
        "    assert name not in sys.modules, name\n"
    )
    done = subprocess.run(
        [str(python), "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
