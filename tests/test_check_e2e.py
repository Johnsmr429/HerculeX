"""The suite runner: each file in a process of its own, its counts, time and peak memory reported, failures shown."""

from runner import check


def test_a_file_runs_alone_and_reports_what_it_did(tmp_path):
    passing, failing = tmp_path / "test_passing.py", tmp_path / "test_failing.py"
    passing.write_text("import pytest\n\ndef test_a():\n    pass\n\n@pytest.mark.skip\ndef test_b():\n    pass\n")
    failing.write_text("def test_c():\n    assert 1 == 2\n")
    good, bad = check.run_file(passing, []), check.run_file(failing, [])
    assert good["code"] == 0 and good["counts"] == {"passed": 1, "skipped": 1} and good["peak"] > 0.01
    assert bad["code"] == 1 and bad["counts"] == {"failed": 1} and "assert 1 == 2" in bad["output"]
    nothing = check.run_file(passing, ["-k", "nomatch"])
    assert nothing["code"] == check.NO_TESTS
