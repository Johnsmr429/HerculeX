#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
case "${1:-}" in
    "" | -h | --help)
        printf '%s\n' 'Usage: ./run.sh setup' '       ./run.sh unwrap SCROLL[:SCAN] ...' '       ./run.sh MODULE [ARGS]' \
            '       ./run.sh list' '       ./run.sh board --help' '       ./run.sh check [cpu|ports|lint]'
        exit 0
        ;;
    setup)
        lane=${HERCULEX_LANE:-$([ "$(uname)" = Darwin ] && echo apple || echo cuda)}
        command -v uv >/dev/null || {
            printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
            exit 1
        }
        UV_PROJECT_ENVIRONMENT=runtime/python uv sync --locked --extra "$lane" || exit 1
        exec ./run.sh ink.model.weights fetch
        ;;
esac
python=runtime/python/bin/python
if [ ! -x "$python" ]; then
    printf '%s\n' 'HerculeX Python is missing. Run ./run.sh setup' >&2
    exit 1
fi
export TMPDIR="$PWD/runtime/scratch"
mkdir -p "$TMPDIR" || exit 1
export PYTHONDONTWRITEBYTECODE=1
export MallocLargeCache=0
module=$1
shift
case "$module" in
    check) exec "$python" -m runner.check "$@" ;;
    list) exec "$python" -m runner.modules ;;
    board) exec "$python" -m runner.job_board "$@" ;;
    unwrap) exec "$python" -m runner.unwrap "$@" ;;
    *) exec "$python" -m "$module" "$@" ;;
esac
