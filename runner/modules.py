"""Every module ./run.sh can run, by package, with the first line of what it does."""

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SKIPPED = {".venv", "runtime", "data", "ports", "ideas", "backups", "tests", "__pycache__", "node_modules"}


def runnable():
    """(module, summary) of every file with a __main__ block."""
    for path in sorted(HERE.rglob("*.py")):
        if SKIPPED & set(path.relative_to(HERE).parts):
            continue
        tree = ast.parse(path.read_text())
        main = any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test) for node in tree.body)
        if main:
            doc = (ast.get_docstring(tree) or "").strip().split("\n")[0]
            yield ".".join(path.relative_to(HERE).with_suffix("").parts), doc


def main():
    package = None
    for module, doc in runnable():
        top = module.rsplit(".", 1)[0]
        if top != package:
            print(f"\n{top}")
            package = top
        print(f"  {module:42s}{doc}")
    print("\n./run.sh MODULE --help for a module's arguments; ./run.sh check [cpu|ports]; ./run.sh board --help")


if __name__ == "__main__":
    main()
