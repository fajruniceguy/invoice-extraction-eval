"""`.env.example` and the CI workflow stay true to the code they describe."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from docextract.config import Settings
from docextract.ingest.pipeline import IngestSettings

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

spec = importlib.util.spec_from_file_location("secret_scan", ROOT / "tools" / "secret_scan.py")
assert spec is not None and spec.loader is not None
scan = importlib.util.module_from_spec(spec)
sys.modules["secret_scan"] = scan
spec.loader.exec_module(scan)

NAME = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=", re.MULTILINE)


def names_in_example() -> set[str]:
    return set(NAME.findall(EXAMPLE.read_text(encoding="utf-8")))


class TestEnvExample:
    def test_it_names_every_setting_the_code_reads(self) -> None:
        wanted = {name.upper() for name in Settings.model_fields}
        wanted |= {f"INGEST_{name.upper()}" for name in IngestSettings.model_fields}
        assert wanted - names_in_example() == set()

    def test_it_loads_as_is_and_the_defaults_it_documents_are_the_ones_in_force(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in Settings.model_fields:
            monkeypatch.delenv(name.upper(), raising=False)
        settings = Settings(_env_file=EXAMPLE)
        assert settings.llm_model == "deepseek/deepseek-v4-flash-vision-exp"
        assert settings.llm_structured_mode == "json_object"
        assert settings.llm_reasoning_effort == "medium"
        assert settings.llm_price_input_per_mtok > 0 and settings.llm_price_output_per_mtok > 0
        assert settings.max_run_cost_usd is not None and settings.max_run_cost_usd > 0
        assert settings.prelabel_model != settings.llm_model  # the pre-labeler must differ

    def test_every_secret_slot_holds_an_obvious_placeholder(self) -> None:
        text = EXAMPLE.read_text(encoding="utf-8")
        slots = re.findall(
            r"^[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*=(.*)$", text, re.MULTILINE
        )
        assert slots, "the example should show where the keys go"
        for value in slots:
            assert value.strip() == "" or scan.is_placeholder(value.strip()), value

    def test_the_scanner_finds_nothing_in_it(self) -> None:
        text = EXAMPLE.read_text(encoding="utf-8")
        assert [hit for line in text.splitlines() for hit in scan.scan_line(line)] == []

    def test_it_is_not_the_real_env_file_and_it_is_the_only_committed_example(self) -> None:
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "*.env", ".env", "*.env.*"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        assert set(tracked) <= {".env.example"}, tracked

    def test_the_previous_example_from_history_passes_the_scanner_too(self) -> None:
        shown = subprocess.run(
            ["git", "-C", str(ROOT), "show", "3bbd99c:.env.example"],
            capture_output=True,
            text=True,
            check=False,
        )
        if shown.returncode != 0:
            pytest.skip("the history is not available (shallow clone)")
        assert [hit for line in shown.stdout.splitlines() for hit in scan.scan_line(line)] == []


class TestWorkflow:
    def workflow(self) -> dict:  # type: ignore[type-arg]
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def commands(self) -> list[str]:
        (job,) = self.workflow()["jobs"].values()
        return [step["run"] for step in job["steps"] if "run" in step]

    def test_it_runs_on_push_and_pull_request(self) -> None:
        triggers = self.workflow()[
            True if True in self.workflow() else "on"
        ]  # YAML reads `on` as True
        assert set(triggers) == {"push", "pull_request"}

    def test_it_runs_exactly_the_gate_this_repository_documents(self) -> None:
        commands = self.commands()
        assert "python -m uv sync" in commands
        assert "python -m uv run ruff check" in commands
        assert 'python -m uv run pytest -q -m "slow or not slow"' in commands
        assert "python tools/secret_scan.py --history" in commands

    def test_the_documented_gate_in_claude_md_is_the_same_command(self) -> None:
        text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        assert 'python -m uv run pytest -q -m "slow or not slow"' in text

    def test_the_secret_scan_can_see_the_whole_history_and_the_job_uses_python_311(self) -> None:
        (job,) = self.workflow()["jobs"].values()
        checkout = next(step for step in job["steps"] if "checkout" in step.get("uses", ""))
        assert checkout["with"]["fetch-depth"] == 0
        python = next(step for step in job["steps"] if "setup-python" in step.get("uses", ""))
        assert str(python["with"]["python-version"]) == "3.11"

    def test_it_references_no_secret_and_runs_on_windows(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "secrets." not in text and "LLM_API_KEY" not in text
        assert "windows-latest" in text
