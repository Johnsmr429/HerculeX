#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
case "${1:-}" in
    ""|-h|--help)
        printf '%s\n' 'Usage: ./run.sh MODULE [ARGS]' '       ./run.sh runner.pipeline --help' '       ./run.sh board --help' '       ./run.sh check [cpu]'
        exit 0
        ;;
esac
python=runtime/python/bin/python
if [ ! -x "$python" ]; then
    printf '%s\n' 'HerculeX Python is missing. Run UV_PROJECT_ENVIRONMENT=runtime/python uv sync --locked --extra ink' >&2
    exit 1
fi
export TMPDIR="$PWD/runtime/scratch"
mkdir -p "$TMPDIR" || exit 1
export PYTHONDONTWRITEBYTECODE=1
export MallocLargeCache=0
module=$1
shift
case "$module" in
    unwrapping.detection.stage) module=unwrapping.detection.detect_nodes ;;
    unwrapping.level.stage) module=unwrapping.level.run_levels ;;
esac
case "$module" in
    check)
        case "${1:-}" in
            cpu) exec "$python" -m pytest -q -p no:cacheprovider tests --ignore tests/test_tile_reader_e2e.py -k "not gpu" ;;
            "") exec "$python" -m pytest -q -p no:cacheprovider tests ;;
            *) printf '%s\n' 'Usage: ./run.sh check [cpu]' >&2; exit 2 ;;
        esac
        ;;
    board) exec "$python" -m runner.job_board "$@" ;;
    *) exec "$python" -m "$module" "$@" ;;
esac
