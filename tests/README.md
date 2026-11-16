# tests

This folder holds two kinds of test.

```text
*_e2e.py      a stage or tool run whole on synthetic inputs with known truth: a crushed scroll of known windings, a spiral, a small server, a board of its own
*_golden.py   a fast path against a trusted reference: Metal against C, network simplex against an LP solver

./run.sh board room 6 && ./run.sh check    all, about 3 minutes
./run.sh check cpu                         without the GPU and Neural Engine
```

`test_tile_reader_e2e.py` needs Reader v3 (`./run.sh ink.model.weights fetch`) and the GPU.
