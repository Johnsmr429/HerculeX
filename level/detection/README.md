# level/detection

Stage 2 turns CT bricks into sheet nodes. HerculeX processes each 8-voxel cell in one GPU threadgroup (a Metal
threadgroup on the apple lane, a CUDA block on the cuda lane): it computes the structure tensor, takes the CT profile
along the sheet normal, and keeps one node for each sheet peak.

| File | Description |
| --- | --- |
| `detect_nodes.c` | Runs the stage. It fetches chunks of 12 brick rows, finds their nodes, and packs them into `nodes/rows-*`, fetching the next chunk while it processes the current one. |
| `node_stream.h` | Streams brick slabs to either detector and writes nodes in cell order. |
| `nodes_apple/` | The Apple lane's `nodes_apple/kernel.metal` and Objective-C host `nodes_apple/host.m`. |
| `nodes_cuda/` | The CUDA lane's `nodes_cuda/kernel.cu` and C host `nodes_cuda/host.c`, which compiles the kernel with NVRTC. |
| `node_kernels.c` | Find the same nodes on the CPU. This is the reference that the GPU path is tested against. |
| `same_output_methods.txt` | Earlier equal-output digests, optionally prefixed by lane. |

```sh
./run.sh level.pipeline run sheets [SCROLL ...]               # the pipeline stage
./run.sh level.detection.detect_nodes run [SCROLL ...] [--band] [--scan ID]
./run.sh level.detection.detect_nodes store CT_STORE OUT FIRST LAST
```

The stage writes each scan's nodes to `unwrapping/nodes/SCROLL/SCAN/nodes/` in the outputs. It fetches the CT into
`unwrapping/detection-work/` and removes each chunk's CT once its nodes are written.

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_FETCH_CONNECTIONS` | `16` | Sets the number of HTTP connections for fetching CT. |
| `HERCULEX_UNIT_ROWS` | `12` | Sets the number of brick rows owned by each detection unit. |
| `HERCULEX_LIST_CT_BRICKS` | unset | Set to `1` to skip bricks absent from a complete bucket listing. Incomplete listings fall back to individual requests. |
