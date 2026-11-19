# unwrapping/detection

Stage 2 turns CT bricks into sheet nodes. HerculeX processes each 8-voxel cell in one GPU threadgroup (a Metal
threadgroup on the apple lane, a CUDA block on the cuda lane): it computes the structure tensor, takes the CT profile
along the sheet normal, and keeps one node for each sheet peak.

| File | Description |
| --- | --- |
| `detect_nodes.py` | Runs the stage. It fetches chunks of 12 brick rows, finds their nodes, and packs them into `nodes/rows-*`, fetching the next chunk while it processes the current one. |
| `gpu_nodes.py` | Finds the nodes on the GPU with the lane's kernel, reading the next slab while the GPU works on this one. |
| `nodes_apple/` | The apple lane's kernel: `kernel.metal`, its Objective-C host `host.m`, and `library.py`, which builds the host. |
| `nodes_cuda/` | The cuda lane's kernel: `kernel.cu`, line for line the Metal kernel, its C host `host.c`, and `library.py`. The host compiles the kernel at first use with NVRTC, which PyTorch's CUDA wheels carry, so no CUDA toolkit is needed. |
| `cpu_nodes.py`, `node_kernels.c` | Find the same nodes on the CPU. This is the reference that the GPU path is tested against. |
| `kernel_bindings.py` | Opens the lane's kernel and the C reference, and defines the ctypes signatures and the layout of the node record. |
| `same_output_methods.txt` | Lists earlier code digests that are proven to give the same nodes. |

```sh
./run.sh runner.pipeline run sheets [SCROLL ...]               # the pipeline stage
./run.sh unwrapping.detection.detect_nodes run [SCROLL ...]     # the same stage, called directly
```

The stage writes each scan's nodes to `unwrapping/nodes/SCROLL/SCAN/nodes/` in the outputs. It fetches the CT into
`unwrapping/detection-work/` and removes each chunk's CT once its nodes are written.

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_FETCH_CONNECTIONS` | `16` | Sets the number of HTTP connections for fetching CT. |
| `HERCULEX_LIST_CT_BRICKS` | unset | Set to `1` to list the published bricks of each chunk first and skip masked-out bricks. |
