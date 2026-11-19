"""Every import of HerculeX's own modules, at the top of a file or inside a function, names a module and names that exist."""

import ast
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
SKIPPED = {".venv", "runtime", "data", "ports", "ideas", "backups", "__pycache__", "node_modules"}
PACKAGES = {path.parent.name for path in HERE.glob("*/__init__.py")} | {"ideas"}


def sources():
    for path in sorted(HERE.rglob("*.py")):
        if not SKIPPED & set(path.relative_to(HERE).parts):
            yield path


def module_file(name):
    """The file of a module of this checkout, or None."""
    base = HERE.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def top_level_names(path):
    """Names a module defines at its top level, including in if, try and with blocks."""
    names, todo = set(), list(ast.parse(path.read_text()).body)
    while todo:
        node = todo.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.For | ast.With):
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            targets += [item.optional_vars for item in getattr(node, "items", []) if item.optional_vars]
            names |= {n.id for t in targets if t is not None for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        if isinstance(node, ast.If | ast.Try | ast.With | ast.For):
            todo += [*node.body, *getattr(node, "orelse", []), *getattr(node, "finalbody", [])]
            todo += [n for handler in getattr(node, "handlers", []) for n in handler.body]
    return names


def broken_imports(path):
    """Each own import in a file that names a missing module or a name its module does not define."""
    where = path.relative_to(HERE) if path.is_relative_to(HERE) else path.name
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PACKAGES and module_file(alias.name) is None:
                    yield f"{where}:{node.lineno} import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] not in PACKAGES:
                continue
            target = module_file(node.module)
            if target is None:
                yield f"{where}:{node.lineno} from {node.module}"
                continue
            defined = top_level_names(target)
            for alias in node.names:
                if alias.name != "*" and alias.name not in defined and not module_file(f"{node.module}.{alias.name}"):
                    yield f"{where}:{node.lineno} from {node.module} import {alias.name}"


def test_every_own_import_resolves():
    broken = [line for path in sources() for line in broken_imports(path)]
    assert not broken, "\n".join(broken)


def test_a_missing_module_or_name_is_caught(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from storage.data_roots import LIBRARY, NO_SUCH_NAME\n"
        "def later():\n"
        "    from storage.no_such_module import thing\n"
        "    import runner.no_such\n"
    )
    found = list(broken_imports(probe))
    assert [line.split(" ", 1)[1] for line in found] == [
        "from storage.data_roots import NO_SUCH_NAME",
        "from storage.no_such_module",
        "import runner.no_such",
    ]


def local_only():
    """The paths release.sh leaves out of the public copy."""
    script = (HERE / "release.sh").read_text()
    listed = script.split("local_only=(", 1)[1].split(")", 1)[0].split()
    kept = script.split("public_datasets=(", 1)[1].split(")", 1)[0].split()
    datasets = [f"ink/datasets/{p.name}" for p in (HERE / "ink" / "datasets").glob("*") if p.name not in kept]
    return [HERE / path for path in [*listed, *datasets]]


def is_local(path, local):
    return any(path == item or item in path.parents for item in local)


def test_no_public_file_imports_one_the_public_copy_leaves_out():
    if not (HERE / "release.sh").exists():
        pytest.skip("the public copy has no release.sh")
    local, shipped = local_only(), []
    for path in sources():
        if is_local(path, local):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                else []
            )
            for name in names:
                target = module_file(name) if name.split(".")[0] in PACKAGES else None
                if target is not None and is_local(target, local):
                    shipped.append(f"{path.relative_to(HERE)}:{node.lineno} imports local {name}")
    assert not shipped, "\n".join(shipped)
