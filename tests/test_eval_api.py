"""scripts/eval_api.py: documents go through POST /extract of an API and come back as the same run
folder `extract.py` writes. The in-process tests drive `evaluate` against the real app over an ASGI
transport with the fake model; the last tests run the real script against a real server process."""

import importlib.util
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from docextract.api.app import create_app
from docextract.config import Settings
from docextract.extract.runner import DocRecord, RunRecord, exit_code, run_extraction
from docextract.ingest.manifest import ManifestRow, write_manifest
from docextract.ingest.pipeline import sha256_hex
from docextract.providers.openai_compat import OpenAICompatClient
from evalkit.groundtruth import GroundTruth
from evalkit.gt_store import write_label
from fakes.fake_llm import FakeLLMServer, Reply
from files import A4, make_sized_pdf
from ingest_support import ROOT, Env, gray_png, settings
from test_api import capped, clean_env, make_settings  # noqa: F401  (clean_env is autouse)
from test_api_server import TOKEN, server  # noqa: F401  (a fixture)
from test_extract_runner import manifest_row
from test_extract_script import scrubbed

sys.path.insert(0, str(ROOT / "scripts"))  # eval_api imports score.py by name, as when run
spec = importlib.util.spec_from_file_location("eval_api", ROOT / "scripts" / "eval_api.py")
assert spec is not None and spec.loader is not None
eval_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_api)


class Dataset:
    def __init__(self, env: Env, tmp_path: Path) -> None:
        self.env = env
        self.labels = tmp_path / "ground_truth"
        self.labels.mkdir()
        self.runs = tmp_path / "runs"
        self.manifest = tmp_path / "manifest.csv"
        self.rows: list[ManifestRow] = []

    def png(self, doc_id: str, *, ingest: bool = True, **overrides: Any) -> ManifestRow:
        row = manifest_row(doc_id, **overrides)
        gray_png(self.env, f"{doc_id}.png")
        if ingest:
            self.env.ingest(row)
        self.rows.append(row)
        write_manifest(self.manifest, self.rows)
        return row

    def pdf(self, doc_id: str, pages: int) -> ManifestRow:
        row = manifest_row(doc_id, raw_files=f"{doc_id}.pdf")
        make_sized_pdf(self.env.raw / f"{doc_id}.pdf", [A4, (420.0, 595.0), (300.0, 300.0)][:pages])
        self.env.ingest(row, settings=settings(dpi=40))
        self.rows.append(row)
        write_manifest(self.manifest, self.rows)
        return row

    def stored(self, doc_id: str) -> list[Path]:
        from docextract.ingest.stored import stored_pages

        return stored_pages(doc_id, self.env.pages)

    async def evaluate(
        self,
        config: Settings,
        *,
        selected: list[ManifestRow] | None = None,
        run_id: str = "r1",
        token: str | None = None,
        concurrency: int = 1,
    ) -> RunRecord:
        async with serve(config, token=token) as http:
            return await eval_api.evaluate(
                self.rows if selected is None else selected,
                self.rows,
                http=http,
                pages_dir=self.env.pages,
                labels_dir=self.labels,
                run_dir=self.runs / run_id,
                selection={"docs": ["x"]},
                concurrency=concurrency,
            )

    def record(self, doc_id: str, run_id: str = "r1") -> DocRecord:
        path = self.runs / run_id / "docs" / f"{doc_id}.json"
        return DocRecord.model_validate_json(path.read_text(encoding="utf-8"))


@asynccontextmanager
async def serve(config: Settings, *, token: str | None = None) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(config)
    async with app.router.lifespan_context(app):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://api.test", headers=headers
        ) as http:
            yield http


@pytest.fixture
def data(env: Env, tmp_path: Path) -> Dataset:
    return Dataset(env, tmp_path)


# ------------------------------------------------------------------ the run folder


class TestRunFolder:
    async def test_it_writes_what_extract_py_writes_and_the_score_accepts_it(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_a")
        data.png("doc_b")
        config = make_settings(fake_llm)
        run = await data.evaluate(config)

        assert run.statuses == {"doc_a": "ok", "doc_b": "ok"}
        assert run.totals.ok == 2 and run.stopped is None and exit_code(run) == 0
        on_disk = RunRecord.model_validate_json(
            (data.runs / "r1" / "run.json").read_text(encoding="utf-8")
        )
        assert on_disk == run
        for doc_id in ("doc_a", "doc_b"):
            record = data.record(doc_id)
            assert record.status == "ok" and record.validated is not None
            assert record.raw_text and record.parsed is not None  # the full record, not compact
        assert len(fake_llm.requests) == 2

    async def test_run_json_describes_the_server_not_the_client(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        from docextract.extract.prompt import SYSTEM_PROMPT, prompt_sha256
        from docextract.extract.schema import DocumentExtraction
        from docextract.providers.cache import schema_hash

        data.png("doc_a")
        config = make_settings(fake_llm, llm_reasoning_effort="low", **capped())
        run = await data.evaluate(config)
        assert run.model == "fake-extractor" and run.endpoint == fake_llm.base_url
        assert run.prompt_sha256 == prompt_sha256(SYSTEM_PROMPT)
        assert run.schema_sha256 == schema_hash(DocumentExtraction)
        assert run.settings["reasoning_effort"] == "low"
        assert run.settings["max_run_cost_usd"] == "1.00"
        # what the server spent during this run, from its own budget
        assert run.budget_spent_usd == Decimal("1.5")  # capped(): every call costs $1.50

    async def test_the_spend_is_what_this_run_added_to_a_server_that_was_already_used(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_a")
        config = make_settings(
            fake_llm, llm_price_input_per_mtok="1000", llm_price_output_per_mtok="0"
        )
        async with serve(config) as http:
            runs = [
                await eval_api.evaluate(
                    data.rows,
                    data.rows,
                    http=http,
                    pages_dir=data.env.pages,
                    labels_dir=data.labels,
                    run_dir=data.runs / name,
                    selection={"docs": ["doc_a"]},
                )
                for name in ("first", "second")
            ]
        assert runs[0].budget_spent_usd > 0
        assert runs[1].budget_spent_usd == runs[0].budget_spent_usd

    async def test_the_settings_it_records_are_the_ones_extract_py_would_record(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_a")
        config = make_settings(fake_llm, llm_reasoning_effort="medium", llm_max_pages=9)
        via_api = await data.evaluate(config)
        async with OpenAICompatClient(config) as client:
            direct = await run_extraction(
                data.rows,
                rows=data.rows,
                client=client,
                settings=config,
                pages_dir=data.env.pages,
                labels_dir=data.labels,
                run_dir=data.runs / "direct",
                selection={"docs": ["x"]},
            )
        assert via_api.settings == direct.settings
        assert via_api.prompt_sha256 == direct.prompt_sha256
        assert via_api.schema_sha256 == direct.schema_sha256
        assert via_api.model == direct.model and via_api.endpoint == direct.endpoint

    async def test_the_pages_of_a_document_are_sent_in_order_and_unchanged(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.pdf("doc_p", 3)
        stored = data.stored("doc_p")
        assert len(stored) == 3 and len({sha256_hex(page.read_bytes()) for page in stored}) == 3
        await data.evaluate(make_settings(fake_llm))
        record = data.record("doc_p")
        assert [ref.sha256 for ref in record.pages] == [
            sha256_hex(page.read_bytes()) for page in stored
        ]
        parts = fake_llm.requests[0].body["messages"][1]["content"]
        assert sum(part["type"] == "image_url" for part in parts) == 3


# ------------------------------------------------------------------ refusals and failures


class TestRefusalsAndFailures:
    async def test_a_blind_document_without_a_label_is_refused_with_no_request(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_x", label_mode="blind", split="test")
        data.png("doc_y")
        run = await data.evaluate(make_settings(fake_llm))
        assert run.statuses == {"doc_x": "refused", "doc_y": "ok"}
        assert "BLIND" in (data.record("doc_x").error or "")
        assert len(fake_llm.requests) == 1  # doc_y only

    async def test_a_blind_document_with_its_label_goes_through(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        from datetime import UTC, datetime

        data.png("doc_x", label_mode="blind", split="test")
        write_label(
            data.labels,
            GroundTruth(
                doc_id="doc_x", label_mode="blind", labeled_at=datetime(2026, 10, 8, tzinfo=UTC)
            ),
        )
        run = await data.evaluate(make_settings(fake_llm))
        assert run.statuses == {"doc_x": "ok"}

    async def test_a_document_that_is_not_ingested_is_refused(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_n", ingest=False)
        run = await data.evaluate(make_settings(fake_llm))
        assert run.statuses == {"doc_n": "refused"}
        assert fake_llm.requests == [] and exit_code(run) == 1

    async def test_a_provider_failure_is_recorded_and_the_run_goes_on(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_a")
        data.png("doc_b")
        fake_llm.enqueue(Reply(500, {"error": {"message": "upstream exploded"}}))
        run = await data.evaluate(make_settings(fake_llm))
        assert run.statuses == {"doc_a": "failed", "doc_b": "ok"}
        failed = data.record("doc_a")
        assert failed.status == "failed" and "upstream exploded" in (failed.error or "")
        assert exit_code(run) == 1

    async def test_the_servers_cost_cap_stops_the_run(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        for name in ("doc_a", "doc_b", "doc_c"):
            data.png(name)
        run = await data.evaluate(make_settings(fake_llm, **capped()))
        assert run.statuses == {"doc_a": "ok", "doc_b": "not_run", "doc_c": "not_run"}
        assert run.stopped == "cost_cap" and exit_code(run) == 3
        assert len(fake_llm.requests) == 1
        assert "cost cap" in (data.record("doc_b").error or "")

    async def test_a_cost_cap_hit_on_a_document_is_attributed_to_it(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        """The first document already crosses the cap; only the ones after it are not_run."""
        data.png("doc_a")
        data.png("doc_b")
        run = await data.evaluate(make_settings(fake_llm, **capped()))
        assert data.record("doc_a").status == "ok"
        assert run.totals.not_run == 1


# ------------------------------------------------------------------ the token and a bad server


class TestServerAccess:
    async def test_the_token_is_sent_as_a_bearer_header(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_a")
        config = make_settings(fake_llm, api_token="tok-123")
        run = await data.evaluate(config, token="tok-123")
        assert run.statuses == {"doc_a": "ok"}

    async def test_a_wrong_token_is_an_unavailable_api_not_a_failed_document(
        self, data: Dataset, fake_llm: FakeLLMServer
    ) -> None:
        data.png("doc_a")
        config = make_settings(fake_llm, api_token="tok-123")
        with pytest.raises(eval_api.ApiUnavailable, match="token"):
            await data.evaluate(config, token="wrong")
        assert fake_llm.requests == []

    async def test_a_server_that_does_not_report_its_hashes_is_refused(self) -> None:
        def old_server(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok", "model": "m"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(old_server), base_url="http://old"
        ) as http:
            with pytest.raises(eval_api.ApiUnavailable, match="prompt_sha256"):
                await eval_api.server_facts(http)

    async def test_an_error_without_a_record_still_says_what_the_api_said(
        self, data: Dataset
    ) -> None:
        data.png("doc_a")

        def broken(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500, json={"error": {"code": "internal", "message": "the disk is full"}}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(broken), base_url="http://broken"
        ) as http:
            record, hit_cap = await eval_api.extract_via_api(http, "doc_a", data.stored("doc_a"))
        assert record.status == "failed" and not hit_cap
        assert "HTTP 500 internal: the disk is full" in (record.error or "")

    async def test_something_that_is_not_json_is_not_our_api(self) -> None:
        def web_page(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>hello</html>")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(web_page), base_url="http://web"
        ) as http:
            with pytest.raises(eval_api.ApiUnavailable, match="not a docextract API"):
                await eval_api.server_facts(http)


# ------------------------------------------------------------------ the command line


class TestMain:
    def argv(self, data: Dataset, *extra: str) -> list[str]:
        return [
            "--manifest",
            str(data.manifest),
            "--pages-dir",
            str(data.env.pages),
            "--ground-truth-dir",
            str(data.labels),
            "--runs-dir",
            str(data.runs),
            *extra,
        ]

    def test_an_unreachable_api_exits_2_and_writes_no_run(
        self, data: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data.png("doc_a")
        code = eval_api.main(
            self.argv(data, "--doc", "doc_a", "--api-url", "http://127.0.0.1:1", "--run-id", "r")
        )
        assert code == 2
        assert "cannot be reached" in capsys.readouterr().err
        assert not (data.runs / "r" / "run.json").exists()

    def test_an_unknown_document_exits_2(
        self, data: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data.png("doc_a")
        assert eval_api.main(self.argv(data, "--doc", "doc_zzz")) == 2
        assert "not in the manifest" in capsys.readouterr().err

    def test_an_existing_run_folder_is_never_overwritten(
        self, data: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data.png("doc_a")
        (data.runs / "r").mkdir(parents=True)
        assert eval_api.main(self.argv(data, "--doc", "doc_a", "--run-id", "r")) == 2
        assert "already exists" in capsys.readouterr().err

    def test_a_bad_run_id_exits_2(self, data: Dataset) -> None:
        data.png("doc_a")
        assert eval_api.main(self.argv(data, "--doc", "doc_a", "--run-id", "../x")) == 2

    def test_dry_run_sends_nothing_and_says_what_it_would_refuse(
        self, data: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data.png("doc_a")
        data.png("doc_x", label_mode="blind", split="test")
        assert eval_api.main(self.argv(data, "--split", "test", "--dry-run")) == 0
        out = capsys.readouterr().out
        assert "would refuse doc_x" in out and "nothing was sent" in out
        assert not data.runs.exists()


# ------------------------------------------------------------------ the real script, a real server


@pytest.mark.slow
class TestAgainstARealServer:
    """The real `scripts/eval_api.py` against `uvicorn --factory` in another process, with an empty
    env file and a scrubbed environment (no key, no .env)."""

    def run_script(self, base: str, data: Dataset, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "eval_api.py"),
                "--api-url",
                base,
                "--manifest",
                str(data.manifest),
                "--pages-dir",
                str(data.env.pages),
                "--ground-truth-dir",
                str(data.labels),
                "--runs-dir",
                str(data.runs),
                *extra,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=scrubbed(),
            cwd=ROOT,
            timeout=180,
        )

    def labeled(self, data: Dataset) -> None:
        from datetime import UTC, datetime

        data.png("doc_a")
        write_label(
            data.labels,
            GroundTruth(
                doc_id="doc_a", label_mode="prelabel", labeled_at=datetime(2026, 10, 8, tzinfo=UTC)
            ),
        )

    def test_it_extracts_through_the_server_and_scores(
        self,
        data: Dataset,
        server: tuple[str, Path],  # noqa: F811
        fake_llm: FakeLLMServer,
    ) -> None:
        base, calls = server
        self.labeled(data)
        done = self.run_script(base, data, "--doc", "doc_a", "--run-id", "e2e", "--token", TOKEN)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "ok        doc_a" in done.stdout
        run_dir = data.runs / "e2e"
        assert (run_dir / "run.json").is_file() and (run_dir / "docs" / "doc_a.json").is_file()
        assert (run_dir / "score" / "scorecard.md").is_file()
        assert len(fake_llm.requests) == 1
        line = json.loads(calls.read_text(encoding="utf-8").splitlines()[0])
        assert line["route"] == "/extract"

    def test_a_run_that_cannot_be_scored_exits_1_even_though_every_document_is_ok(
        self,
        data: Dataset,
        server: tuple[str, Path],  # noqa: F811
        fake_llm: FakeLLMServer,
    ) -> None:
        base, _ = server
        data.png("doc_a")  # no label: nothing to score
        done = self.run_script(base, data, "--doc", "doc_a", "--run-id", "e2e", "--token", TOKEN)
        assert done.returncode == 1, done.stdout + done.stderr
        assert "ok        doc_a" in done.stdout
        assert "no document has a ground-truth label yet" in done.stderr

    def test_without_the_token_it_exits_2(
        self,
        data: Dataset,
        server: tuple[str, Path],  # noqa: F811
        fake_llm: FakeLLMServer,
    ) -> None:
        base, _ = server
        self.labeled(data)
        done = self.run_script(base, data, "--doc", "doc_a", "--run-id", "e2e")
        assert done.returncode == 2
        assert "token" in done.stderr
        assert fake_llm.requests == []
        assert not (data.runs / "e2e" / "run.json").exists()
