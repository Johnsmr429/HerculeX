# unwrapping/detection

Stage 2: CT bricks to sheet nodes.

```text
detect_nodes.py                                      chunks of 12 brick rows fetched, read and packed into nodes/rows-*
gpu_nodes.py, gpu_nodes.metal, gpu_nodes_host.m   one Metal threadgroup per cell: structure tensor, depth profile, sheet peaks
cpu_nodes.py, node_kernels.c                  the same on the CPU, the reference the GPU is tested against
kernel_bindings.py                            ctypes signatures and the node record layout

./run.sh runner.pipeline run sheets [SCROLL ...]
```
