# tests

The tests run each stage or tool on synthetic inputs with known truth, and check fast paths against trusted
references.

| Files | What they test |
| --- | --- |
| `*_e2e.py` | A stage or tool run whole on synthetic inputs with known truth, such as a crushed scroll of known windings, a spiral, a small HTTP server, or a job board of its own. |
| `*_golden.py` | A fast path against a trusted reference: the GPU sheet-node kernel (Metal, or CUDA off macOS) against the C kernel, the network simplex against a general LP solver, and, off macOS with clang, the `dispatch_apply` side of `parallel_blocks.h`. |
| `test_reflatten.py` | The near-isometric flattening in `spots.reflatten`. |

```sh
./run.sh board room 6 && ./run.sh check    # every test, once 6 GB fits beside the running jobs
./run.sh check cpu                         # only the tests that need neither the GPU nor the Neural Engine
```

`test_tile_reader_e2e.py` needs the GPU and Reader v3; it skips its tests until you run
`./run.sh ink.model.weights fetch`.
