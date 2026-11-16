# Tests

The test suite contains end-to-end tests over synthetic inputs and golden tests against trusted implementations.

| Pattern | Purpose |
| --- | --- |
| `*_e2e.py` | These tests run a complete stage or tool with small inputs and known results. |
| `*_golden.py` | These tests compare an optimized path with a trusted reference implementation. |

Run the CPU-only suite before every change:

```sh
./run.sh check cpu
```

Run `./run.sh check` when a change affects the GPU, Neural Engine, or reader. The tile-reader test requires stock Reader v3; fetch it with `./run.sh ink.model.weights fetch`.
