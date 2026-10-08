"""The secret scanner and the Claude Code hook around it.

Every fake secret is assembled at runtime, so this file itself holds nothing key-shaped (the scanner
runs over the repository's own history in CI)."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scan = load("secret_scan", ROOT / "tools" / "secret_scan.py")
hook = load("pre_commit_secret_scan", ROOT / ".claude" / "hooks" / "pre_commit_secret_scan.py")

SK = "sk-" + "Ab1" * 10
GOOGLE = "AIza" + "Ab1" * 12
GITHUB = "gh" + "p_" + "Ab1" * 12
AWS = "AK" + "IA" + "ABCD1234" * 2
SLACK = "xox" + "b-" + "1234567890-abcdef"
PEM = "-----BEGIN " + "RSA PRIVATE KEY-----"
ASSIGNED = "q9" * 12  # 24 characters, a letter and a digit


class TestRules:
    @pytest.mark.parametrize(
        ("secret", "rule"),
        [
            (SK, "openai-style key"),
            (GOOGLE, "google api key"),
            (GITHUB, "github token"),
            (AWS, "aws access key id"),
            (SLACK, "slack token"),
            (PEM, "private key block"),
        ],
    )
    def test_each_known_key_shape_is_found_and_removed_from_the_excerpt(
        self, secret: str, rule: str
    ) -> None:
        ((found, excerpt),) = scan.scan_line(f"value = '{secret}'  # comment")
        assert found == rule
        assert secret not in excerpt and "[REDACTED" in excerpt and "# comment" in excerpt

    def test_a_secret_assigned_to_a_secret_name(self) -> None:
        ((rule, excerpt),) = scan.scan_line(f"MY_API_KEY={ASSIGNED}")
        assert rule == "secret assigned to MY_API_KEY"
        assert ASSIGNED not in excerpt and excerpt.startswith("MY_API_KEY=[REDACTED 24 chars]")

    @pytest.mark.parametrize("name", ["api_token", "DB_PASSWORD", "client-secret", "x.api.key"])
    def test_the_name_may_be_any_spelling_of_key_token_secret_password(self, name: str) -> None:
        assert scan.scan_line(f'{name}: "{ASSIGNED}"')

    @pytest.mark.parametrize(
        "line",
        [
            "LLM_API_KEY=your-ai-studio-api-key",
            "PRELABEL_API_KEY=your-anthropic-api-key",
            "API_TOKEN=",
            "SOME_KEY=" + "x" * 24,
            "LLM_API_KEY=<paste-your-key-1234567890>",
            "api_key = ${API_KEY_FROM_ENV_1234}",
            "SECRET=" + "a" * 20,
            "KEY=put-the-real-value-here-1234567890",
            "LLM_API_KEY=sk-...",
        ],
    )
    def test_obvious_placeholders_are_silent(self, line: str) -> None:
        assert scan.scan_line(line) == []

    @pytest.mark.parametrize(
        "line",
        [
            "LLM_MAX_PAGES=20",
            "model = 'a-model-with-a-long-name-1234'",
            "api_key = settings.llm_api_key_for_this_client",
            "tokenizer_path = models/tokenizer/long_name_here_2",
            "max_tokens = 8000",
            "request_key = " + "1234567890" * 2,
            "password = hunter2",
            "secret_name = a_long_identifier_without_digits",
            "url = 'https://example.com/v1/keys/abcdefghijklmnop1234'",
        ],
    )
    def test_code_and_config_that_only_mention_the_words_are_silent(self, line: str) -> None:
        assert scan.scan_line(line) == []

    def test_a_key_shape_in_an_assignment_is_one_finding_not_two(self) -> None:
        assert len(scan.scan_line(f"LLM_API_KEY={SK}")) == 1

    def test_two_secrets_in_one_line_are_two_findings(self) -> None:
        assert len(scan.scan_line(f"{SK} {GOOGLE}")) == 2


class TestLongLines:
    """A minified file is one enormous line, so the scan must stay linear in it. An unbounded NAME
    prefix once made it quadratic: a 100 KB line stalled the scan (a mutation pass that hung)."""

    SECRET = "MY_API_KEY=" + ASSIGNED

    def timed(self, line: str) -> tuple[list[tuple[str, str]], float]:
        started = time.perf_counter()
        found = scan.scan_line(line)
        return found, time.perf_counter() - started

    @pytest.mark.parametrize("unit", ["x", "a1", "a-b_c.d"])
    def test_two_megabytes_without_a_keyword_cost_next_to_nothing(self, unit: str) -> None:
        found, seconds = self.timed(unit * (2_000_000 // len(unit)))
        assert found == [] and seconds < 5

    def test_a_secret_before_or_after_a_huge_run_is_still_found(self) -> None:
        for line in (self.SECRET + " " + "x" * 1_000_000, "x" * 1_000_000 + " " + self.SECRET):
            found, seconds = self.timed(line)
            assert [rule for rule, _ in found] == ["secret assigned to MY_API_KEY"]
            assert seconds < 5

    def test_a_line_made_of_keywords_is_bounded_too(self) -> None:
        found, seconds = self.timed("key " * 200_000)
        assert found == [] and seconds < 5

    def test_a_very_long_name_is_still_caught_by_its_last_part(self) -> None:
        found, _ = self.timed("x" * 500 + f"_API_KEY={ASSIGNED}")
        assert len(found) == 1 and ASSIGNED not in found[0][1]


DIFF = "\n".join(
    [
        "diff --git a/app.py b/app.py",
        "--- a/app.py",
        "+++ b/app.py",
        "@@ -1,2 +1,2 @@",
        " context line " + SK,
        f"-removed {SK}",
        "+added clean line",
        "@@ -10,0 +11,2 @@",
        "+first",
        f"+KEY = '{GOOGLE}'",
    ]
)


class TestDiff:
    def test_only_added_lines_count_and_the_line_number_is_the_new_one(self) -> None:
        found = list(scan.scan_diff(DIFF))
        assert [(f.where, f.rule) for f in found] == [("app.py:12", "google api key")]

    def test_context_and_removed_lines_advance_the_numbering_correctly(self) -> None:
        diff = "\n".join(["+++ b/a.txt", "@@ -1,3 +5,3 @@", " same", "-gone", "+new", f"+{SK}"])
        assert [f.where for f in scan.scan_diff(diff)] == ["a.txt:7"]

    def test_the_prefix_names_the_commit(self) -> None:
        (found,) = scan.scan_diff(DIFF, prefix="commit abc ")
        assert found.where == "commit abc app.py:12"

    @pytest.mark.parametrize("path", [".env", "config/.env", "prod.env", "a/b/dev.env"])
    def test_an_env_file_is_refused_whatever_it_holds(self, path: str) -> None:
        (found,) = scan.scan_diff(f"+++ b/{path}\n@@ -0,0 +1 @@\n+HARMLESS=1")
        assert found.rule == "env file added"

    def test_the_example_file_and_deleted_files_are_fine(self) -> None:
        assert (
            list(scan.scan_diff("+++ b/.env.example\n@@ -0,0 +1 @@\n+LLM_API_KEY=your-key")) == []
        )
        assert list(scan.scan_diff("+++ /dev/null\n@@ -1 +0,0 @@\n-" + SK)) == []


class Repo:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q")
        self.git("config", "user.email", "t@example.invalid")
        self.git("config", "user.name", "t")

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(self.path), *args], capture_output=True, text=True, check=True
        )
        return done.stdout

    def write(self, name: str, text: str) -> Path:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def commit(self, name: str, text: str, message: str = "m") -> None:
        self.write(name, text)
        self.git("add", name)
        self.git("commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path / "r")


class TestStagedAndWorktree:
    def test_the_index_is_scanned_and_the_report_never_holds_the_secret(self, repo: Repo) -> None:
        repo.write("settings.py", f"line one\nKEY = '{SK}'\n")
        repo.git("add", "settings.py")
        (found,) = scan.staged_findings(repo.path)
        assert (found.where, found.rule) == ("settings.py:2", "openai-style key")
        assert SK not in str(found)

    def test_a_secret_already_committed_is_not_reported_again_for_an_unrelated_change(
        self, repo: Repo
    ) -> None:
        repo.commit("a.py", f"KEY = '{SK}'\nother = 1\n")
        repo.write("a.py", f"KEY = '{SK}'\nother = 2\n")
        repo.git("add", "a.py")
        assert scan.staged_findings(repo.path) == []

    def test_a_staged_env_file_is_refused_and_the_example_is_not(self, repo: Repo) -> None:
        repo.write(".env", "HARMLESS=1\n")
        repo.write(".env.example", "LLM_API_KEY=your-key-here\n")
        repo.git("add", "-f", ".env", ".env.example")
        assert [f.rule for f in scan.staged_findings(repo.path)] == ["env file added"]

    def test_the_worktree_adds_unstaged_edits_and_untracked_files(self, repo: Repo) -> None:
        repo.commit("a.py", "x = 1\n")
        repo.write("a.py", f"x = 1\nKEY = '{SK}'\n")  # unstaged edit
        repo.write("new.py", f"TOKEN = '{GOOGLE}'\n")  # untracked
        found = {f.where: f.rule for f in scan.worktree_findings(repo.path)}
        assert found == {"a.py:2": "openai-style key", "new.py:1": "google api key"}
        assert scan.staged_findings(repo.path) == []  # nothing is staged

    def test_untracked_binary_and_huge_files_are_skipped(self, repo: Repo) -> None:
        (repo.path / "blob.bin").write_bytes(b"\0" + SK.encode())
        (repo.path / "big.txt").write_text(SK + " " + "x" * (scan.MAX_UNTRACKED_BYTES + 1))
        assert scan.worktree_findings(repo.path) == []

    def test_an_untracked_env_file_is_refused_but_a_gitignored_one_is_invisible(
        self, repo: Repo
    ) -> None:
        repo.commit(".gitignore", ".env\n")
        repo.write(".env", f"LLM_API_KEY={SK}\n")
        assert scan.worktree_findings(repo.path) == []  # ignored: never part of a commit
        repo.write("prod.env", "A=1\n")
        assert [f.rule for f in scan.worktree_findings(repo.path)] == ["env file added"]


class TestHistory:
    def test_a_secret_added_and_removed_again_is_still_in_the_history(self, repo: Repo) -> None:
        repo.commit("a.py", f"KEY = '{SK}'\n", "add")
        repo.commit("a.py", "KEY = None\n", "remove")
        found = scan.history_findings(repo.path)
        first = repo.git("rev-list", "--max-parents=0", "HEAD").strip()
        assert [(f.where, f.rule) for f in found] == [
            (f"commit {first[:10]} a.py:1", "openai-style key")
        ]
        assert SK not in str(found[0])

    def test_every_ref_is_scanned_not_only_the_current_branch(self, repo: Repo) -> None:
        repo.commit("a.py", "x = 1\n")
        repo.git("checkout", "-q", "-b", "other")
        repo.commit("b.py", f"KEY = '{GOOGLE}'\n")
        repo.git("checkout", "-q", "-")
        assert [f.rule for f in scan.history_findings(repo.path)] == ["google api key"]

    def test_a_clean_history_is_clean(self, repo: Repo) -> None:
        repo.commit("a.py", "x = 1\n")
        assert scan.history_findings(repo.path) == []


class TestEnvValues:
    def test_a_value_from_the_real_env_found_in_history_is_reported_by_name_only(
        self, repo: Repo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo.commit("notes.txt", f"I pasted {ASSIGNED} here by mistake\n", "oops")
        repo.write(".env", f"FAKE_API_KEY={ASSIGNED}\nLLM_MAX_PAGES=20\n")
        code = scan.main(["--env-values", "--repo", str(repo.path)])
        out = capsys.readouterr()
        assert code == 1
        assert "FAKE_API_KEY" in out.out and "checking 1 value(s)" in out.out
        assert ASSIGNED not in out.out and ASSIGNED not in out.err

    def test_a_key_made_only_of_letters_is_still_searched(
        self, repo: Repo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        letters = "qwertyuiopasdfghjklzxcvbnm"
        repo.commit("n.txt", f"{letters}\n")
        repo.write(".env", f"MY_TOKEN={letters}\n")
        assert scan.main(["--env-values", "--repo", str(repo.path)]) == 1
        assert letters not in capsys.readouterr().out

    def test_placeholders_short_values_and_non_secret_names_are_not_searched(
        self, repo: Repo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo.commit("n.txt", "your-key-here short123 20\n")
        repo.write(".env", "LLM_API_KEY=your-key-here\nAPI_TOKEN=short\nLLM_MAX_PAGES=20\n")
        assert scan.main(["--env-values", "--repo", str(repo.path)]) == 0
        assert "checking 0 value(s)" in capsys.readouterr().out

    @pytest.mark.parametrize("value", ["z" * 24, "$" + "OTHER_SECRET_1234", "changeme-" + "9" * 12])
    def test_repeated_characters_and_variable_references_are_not_searched_either(
        self, repo: Repo, capsys: pytest.CaptureFixture[str], value: str
    ) -> None:
        repo.commit("n.txt", f"{value}\n")
        repo.write(".env", f"LLM_API_KEY={value}\n")
        assert scan.main(["--env-values", "--repo", str(repo.path)]) == 0
        assert "checking 0 value(s)" in capsys.readouterr().out

    def test_no_env_file_is_a_usage_error(self, repo: Repo) -> None:
        assert scan.main(["--env-values", "--repo", str(repo.path)]) == 2


class TestMain:
    def test_exit_codes(self, repo: Repo, capsys: pytest.CaptureFixture[str]) -> None:
        repo.write("a.py", "x = 1\n")
        repo.git("add", "a.py")
        assert scan.main(["--staged", "--repo", str(repo.path)]) == 0
        repo.write("b.py", f"KEY = '{SK}'\n")
        repo.git("add", "b.py")
        assert scan.main(["--staged", "--repo", str(repo.path)]) == 1
        out = capsys.readouterr().out
        assert "b.py:1: openai-style key" in out and "1 finding(s) in the staged changes" in out

    def test_not_a_repository_is_exit_2_not_a_clean_bill(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "plain").mkdir()
        assert scan.main(["--history", "--repo", str(tmp_path / "plain")]) == 2
        assert "secret_scan:" in capsys.readouterr().err

    def test_one_mode_is_required(self) -> None:
        with pytest.raises(SystemExit) as stopped:
            scan.main([])
        assert stopped.value.code == 2


class TestHookPlan:
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("git commit -m 'x'", (True, False, None)),
            ('git commit -m "a && b"', (True, False, None)),
            ("git add a.py && git commit -m x", (True, True, None)),
            ("git add -A; git commit -m x", (True, True, None)),
            ("git commit -am x", (True, True, None)),
            ("git commit -a -m x", (True, True, None)),
            ("git commit --all -m x", (True, True, None)),
            ("git -C sub commit -m x", (True, False, "sub")),
            ("git.exe commit -m x", (True, False, None)),
            ('"C:/Program Files/Git/cmd/git.exe" add . && git.exe commit -m x', (True, True, None)),
            ("git -c user.name=a commit -m x", (True, False, None)),
            ("pytest -q && git commit -m x", (True, False, None)),
            ("git status", (False, False, None)),
            ("git log --grep commit", (False, False, None)),
            ("git add a.py", (False, False, None)),
            ("echo git commit", (False, False, None)),
            ("python -m uv run pytest", (False, False, None)),
            ("", (False, False, None)),
        ],
    )
    def test_which_commands_are_commits_and_what_they_stage(
        self, command: str, expected: tuple[bool, bool, str | None]
    ) -> None:
        assert hook.plan(command) == expected


def run_hook(cwd: Path, command: str, raw: str | None = None) -> subprocess.CompletedProcess[str]:
    payload = (
        raw
        if raw is not None
        else json.dumps({"tool_input": {"command": command}, "cwd": str(cwd)})
    )
    return subprocess.run(
        [sys.executable, str(ROOT / ".claude" / "hooks" / "pre_commit_secret_scan.py")],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


@pytest.mark.slow
class TestHook:
    def staged_secret(self, repo: Repo) -> None:
        repo.write("leak.py", f"KEY = '{SK}'\n")
        repo.git("add", "leak.py")

    def test_a_commit_with_a_staged_secret_is_blocked_and_says_why(self, repo: Repo) -> None:
        self.staged_secret(repo)
        done = run_hook(repo.path, "git commit -m x")
        assert done.returncode == 2
        assert "Blocked" in done.stderr and "leak.py:1: openai-style key" in done.stderr
        assert SK not in done.stderr and SK not in done.stdout

    def test_a_clean_commit_passes(self, repo: Repo) -> None:
        repo.write("ok.py", "x = 1\n")
        repo.git("add", "ok.py")
        assert run_hook(repo.path, "git commit -m x").returncode == 0

    def test_other_commands_pass_even_with_a_secret_staged(self, repo: Repo) -> None:
        self.staged_secret(repo)
        for command in ("git status", "git log --grep commit", "echo git commit", "ls"):
            assert run_hook(repo.path, command).returncode == 0, command

    def test_add_and_commit_in_one_command_sees_files_not_yet_staged(self, repo: Repo) -> None:
        repo.write("leak.py", f"KEY = '{SK}'\n")  # untracked: the hook runs before `git add`
        assert run_hook(repo.path, "git add leak.py && git commit -m x").returncode == 2
        assert run_hook(repo.path, "git commit -m x").returncode == 0  # nothing is staged

    def test_commit_dash_a_sees_unstaged_edits_to_tracked_files(self, repo: Repo) -> None:
        repo.commit("a.py", "x = 1\n")
        repo.write("a.py", f"x = 1\nKEY = '{GOOGLE}'\n")
        assert run_hook(repo.path, "git commit -am x").returncode == 2

    def test_dash_c_names_the_repository_to_scan(self, repo: Repo, tmp_path: Path) -> None:
        self.staged_secret(repo)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        assert run_hook(elsewhere, f'git -C "{repo.path}" commit -m x').returncode == 2

    def test_a_malformed_payload_or_a_directory_that_is_not_a_repo_never_blocks(
        self, tmp_path: Path
    ) -> None:
        assert run_hook(tmp_path, "", raw="not json").returncode == 0
        assert run_hook(tmp_path, "", raw="[]").returncode == 0  # valid JSON, but not an object
        plain = tmp_path / "plain"
        plain.mkdir()
        assert run_hook(plain, "git commit -m x").returncode == 0

    def test_the_settings_file_registers_the_hook_on_bash(self) -> None:
        settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
        (entry,) = settings["hooks"]["PreToolUse"]
        assert entry["matcher"] == "Bash"
        (command,) = entry["hooks"]
        assert command["type"] == "command"
        assert "pre_commit_secret_scan.py" in command["command"]
        assert (ROOT / ".claude" / "hooks" / "pre_commit_secret_scan.py").is_file()
