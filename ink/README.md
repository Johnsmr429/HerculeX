# ink

The `ink` package turns each piece's layer stack into an ink map (stage 6). It reads the ink with Reader v3.

| Folder | Contents |
| --- | --- |
| `model/` | Fetches and checks Reader v3 and runs it: the tile reader, with a Metal stem and a Neural Engine body, and a portable PyTorch reader. |
| `reading/` | Runs the ink stage over every piece, and reads circled spots and passages again at other depths. |
| `datasets/` | `labelled_cases.py` collects the published labeled ink segments and infrared fragments that the letter atlas uses. Run `./run.sh ink.datasets.labelled_cases` to build them in the library's `datasets/`. |

## The network

Reader v3 is Domenico Russo's fine-tune of the Vesuvius Challenge `ink_9um` network, released under the MIT license at
<https://huggingface.co/domenicor046/reader-v3>. The network has 34.5 million parameters: a 3D stem over 17 CT layers
feeds a 2D residual U-Net. HerculeX doesn't include model weights. Fetch Reader v3 once before you read ink:

```sh
./run.sh ink.model.weights fetch     # download Reader v3, check it and convert it for the tile reader
./run.sh ink.model.weights status    # show where each checkpoint is and whether it's in place
```

The checkpoint goes to the library's `weights/pretrained/reader_v3/`. HerculeX compiles the Neural Engine build on
first use and keeps it in `weights/engine/`. To read with another checkpoint of the same network, set
`HERCULEX_WEIGHTS` to its path.
