"""Committed files pass their language's pinned linters (runner/lint.py); files with uncommitted changes, which another
session may be editing, are left to their author's own check."""

import subprocess

import pytest

from runner import lint


def committed():
    """Whether a tracked file has no uncommitted change."""
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--relative", "HEAD", "--", "."],
        cwd=lint.HERE,
        capture_output=True,
        text=True,
        check=False,
    )
    if changed.returncode:
        pytest.skip("not a git checkout")
    edited = set(changed.stdout.split())
    return lambda path: path not in edited


CHECKS = dict(lint.checks())


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_committed_files_pass_their_linters(name):
    command = dict(lint.checks(committed())).get(name)
    if command is None:
        pytest.skip("every file it checks has uncommitted changes")
    found = subprocess.run(command, cwd=lint.HERE, capture_output=True, text=True, check=False)
    assert found.returncode == 0, found.stdout + found.stderr
