"""tools/mutation_check.py on a tiny project built in a temp directory. The tool starts pytest
subprocesses, so these tests are slow."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mutation_check", ROOT / "tools" / "mutation_check.py"
)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["mutation_check"] = tool
spec.loader.exec_module(tool)

CALC = "\n".join(
    ["def add(a, b):", "    return a + b", "", ""]
    + ["def sub(a, b):", "    return a - b", "", ""]
    + ["def twice(a):", "    return a + a", ""]
)
TESTS = (
    "import pytest\nfrom calc import add, twice\n\n\n"
    "def test_add():\n    assert add(1, 2) == 3\n\n\n"
    "@pytest.mark.slow\ndef test_twice_only_a_slow_test_checks():\n    assert twice(4) == 8\n"
)
PYPROJECT = '[tool.pytest.ini_options]\naddopts = "-m \'not slow\'"\nmarkers = ["slow: slow"]\n'


def project(tmp_path: Path, *, broken: bool = False) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "calc.py").write_text(CALC, encoding="utf-8")
    body = TESTS + ("\n\ndef test_broken():\n    assert False\n" if broken else "")
    (root / "tests" / "test_calc.py").write_text(body, encoding="utf-8")
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    return root


def mutant(name: str, old: str, new: str, path: str = "src/calc.py") -> "tool.Mutant":
    return tool.Mutant(path, ("tests/test_calc.py",), name, old, new)


def digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verdicts(report: "tool.Report") -> dict[str, str]:
    return {o.name: o.verdict for o in report.outcomes}


COPY = ("src", "tests", "pyproject.toml")


class TestCheck:
    def test_a_mutant_a_test_notices_is_killed_and_one_nothing_notices_survives(
        self, tmp_path: Path
    ) -> None:
        report = tool.check(
            project(tmp_path),
            [
                mutant("add breaks", "return a + b\n\n\ndef sub", "return a - b\n\n\ndef sub"),
                mutant("sub is untested", "return a - b", "return a + b"),
            ],
            copy=COPY,
        )
        assert report.baseline_ok
        assert verdicts(report) == {"add breaks": "killed", "sub is untested": "SURVIVED"}
        assert [o.name for o in report.bad] == ["sub is untested"]

    def test_slow_marked_tests_count_as_tests(self, tmp_path: Path) -> None:
        report = tool.check(
            project(tmp_path), [mutant("twice breaks", "return a + a", "return a * a")], copy=COPY
        )
        assert verdicts(report) == {"twice breaks": "killed"}  # only the slow test can notice

    def test_a_mutant_whose_text_is_missing_or_ambiguous_is_refused_not_run(
        self, tmp_path: Path
    ) -> None:
        report = tool.check(
            project(tmp_path),
            [
                mutant("missing", "return a * b", "return 0"),
                mutant("twice in the file", "return a", "return 0"),
                mutant("no such file", "x", "y", path="src/none.py"),
            ],
            copy=COPY,
        )
        assert verdicts(report) == {
            "missing": "REFUSED",
            "twice in the file": "REFUSED",
            "no such file": "REFUSED",
        }
        assert "0 times" in report.outcomes[0].detail and "3 times" in report.outcomes[1].detail

    def test_a_red_baseline_reports_no_mutant_at_all(self, tmp_path: Path) -> None:
        report = tool.check(
            project(tmp_path, broken=True),
            [mutant("m", "return a + b\n\n\ndef sub", "return 0\n\n\ndef sub")],
            copy=COPY,
        )
        assert not report.baseline_ok and report.outcomes == []
        assert "test_broken" in report.baseline_tail

    def test_the_original_tree_is_never_written(self, tmp_path: Path) -> None:
        root = project(tmp_path)
        before = digest(root)
        tool.check(
            root,
            [mutant("add breaks", "return a + b\n\n\ndef sub", "return 0\n\n\ndef sub")],
            copy=COPY,
        )
        assert digest(root) == before

    def test_the_scratch_copy_is_removed_afterwards(self, tmp_path: Path) -> None:
        import tempfile

        before = {p.name for p in Path(tempfile.gettempdir()).glob("mutation_check_*")}
        tool.check(project(tmp_path), [mutant("m", "return a + a", "return 1")], copy=COPY)
        assert {p.name for p in Path(tempfile.gettempdir()).glob("mutation_check_*")} == before


class TestLoadAndMain:
    def write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "mutants.py"
        path.write_text(body, encoding="utf-8")
        return path

    def test_tests_may_be_one_path_or_a_list(self, tmp_path: Path) -> None:
        loaded = tool.load_mutants(
            self.write(
                tmp_path,
                "MUTANTS = [('a.py', 't.py', 'n', 'o', 'w'),"
                " ('b.py', ['t1', 't2'], 'm', 'o', 'w')]",
            )
        )
        assert [m.tests for m in loaded] == [("t.py",), ("t1", "t2")]

    def test_a_file_without_mutants_or_with_a_malformed_one_is_refused(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(tool.MutationError, match="no MUTANTS"):
            tool.load_mutants(self.write(tmp_path, "x = 1"))
        with pytest.raises(tool.MutationError, match="a mutant is"):
            tool.load_mutants(self.write(tmp_path, "MUTANTS = [('only', 'three', 'fields')]"))

    def test_main_exit_codes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = project(tmp_path)
        item = ("src/calc.py", "tests/test_calc.py", "add")
        item += ("return a + b\n\n\ndef sub", "return 0\n\n\ndef sub")
        killed = self.write(tmp_path, f"MUTANTS = [{item!r}]")
        argv = ["--root", str(root), "--copy", "src,tests,pyproject.toml"]
        assert tool.main([str(killed), *argv]) == 0
        assert "killed     add" in capsys.readouterr().out

        survivor = tmp_path / "survivor.py"
        row = ("src/calc.py", "tests/test_calc.py", "sub", "return a - b", "return a + b")
        survivor.write_text(f"MUTANTS = [{row!r}]", encoding="utf-8")
        assert tool.main([str(survivor), *argv]) == 1
        assert "SURVIVED   sub" in capsys.readouterr().out

        assert tool.main([str(tmp_path / "missing.py"), *argv]) == 2

    def test_main_returns_2_when_the_baseline_is_red(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = project(tmp_path, broken=True)
        mutants = self.write(
            tmp_path, "MUTANTS = [('src/calc.py', 'tests/test_calc.py', 'm', 'a + b', 'a')]"
        )
        code = tool.main([str(mutants), "--root", str(root), "--copy", "src,tests,pyproject.toml"])
        assert code == 2 and "baseline run is red" in capsys.readouterr().err
