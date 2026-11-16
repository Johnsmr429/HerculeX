# ink

Stage 6: layer stacks to ink maps, read with stock Reader v3.

```text
model/      Reader v3 fetched and checked, the tile reader (Metal stem, Neural Engine body), a portable PyTorch reader
reading/    the ink stage over every piece, and re-reads of circled spots at other depths
datasets/   labelled_cases.py: the published labelled segments, for the letter atlas
```

The network: 34.5M parameters, a 3-D stem over 17 layers into a 2-D residual U-Net, the Vesuvius Challenge `ink_9um`
architecture. Reader v3 is its fine-tune by Domenico Russo (MIT, https://huggingface.co/domenicor046/reader-v3).
`./run.sh ink.model.weights fetch` puts it in the library's `weights/pretrained/reader_v3/`; the Neural Engine build is
made on first use in `weights/engine/`. HerculeX ships no ink model of its own. `HERCULEX_WEIGHTS` reads with another
checkpoint of the same network.
