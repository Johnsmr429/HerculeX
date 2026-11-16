# Sheet detection

This stage converts CT bricks into sheet nodes with positions, normals, and cell metadata.

## Files

| File | Purpose |
| --- | --- |
| `detect_nodes.py` | Fetches CT chunks, runs detection, and writes resumable row units. |
| `gpu_nodes.py`, `gpu_nodes.metal`, and `gpu_nodes_host.m` | Process each 8-voxel cell on the GPU and overlap compute with input. |
| `cpu_nodes.py` and `node_kernels.c` | Provide the CPU reference used by the golden tests. |
| `kernel_bindings.py` | Defines the node record layout and typed native calls. |
| `same_output_methods.txt` | Lists earlier method digests with byte-identical outputs. |

## Command

```sh
./run.sh runner.pipeline run sheets [SCROLL ...]
```

The stage writes completed `nodes/rows-*` units beneath each scan in the node route. Run it through the job board with GPU and network resources for a full scroll.
