# tests

The tests run each stage or tool on synthetic inputs with known truth, and check fast paths against trusted
references.

| Files | What they test |
| --- | --- |
| `*_e2e.py` | A stage or tool run whole on synthetic inputs with known truth, such as a crushed scroll of known windings, a spiral, a small HTTP server, or a job board of its own. |
| `*_golden.py` | A fast path against a trusted reference: the lane's GPU sheet-node kernel against the C kernel, the network simplex against a general LP solver, and every level kernel built with both lanes' `parallel_blocks.h`. |
| `test_reflatten.py` | The near-isometric flattening in `spots.reflatten`. |
| `test_lint_e2e.py` | Committed files against the pinned linters of their language, as `./run.sh check lint` runs them. |

```sh
./run.sh board room 6 && ./run.sh check    # every test, once 6 GB fits beside the running jobs
./run.sh check cpu                         # only the tests that need neither the GPU nor the Neural Engine
./run.sh check ports                       # the native ports in ports/, which the pipeline does not call
```

Tests marked `apple` or `cuda` run only on that lane's machines (`tests/conftest.py`). `test_reader_apple_e2e.py` and
`test_reader_cuda_e2e.py` need Reader v3; they skip their tests until you run `./run.sh setup`.

`./run.sh check` runs each test file in a fresh process (`runner/check.py`), three at a time for `cpu`, and ends with
the slowest files and their peak memory. While you work on a module, run only its test files:
`runtime/python/bin/python -m pytest -q tests/test_NAME.py`. Run the whole CPU suite once, before you commit, as a
board job:

```sh
./run.sh board add cpu-check --memory 3 --cores 3 -- ./run.sh check cpu
./run.sh board add full-check --gpu --engine --memory 10 --cores 2 -- ./run.sh check    # GPU memory counts too
```

A test reads data from the library or the outputs only when it can't be made synthetic, and then skips with the
path it needs. A new test builds its inputs itself and finishes in seconds.
