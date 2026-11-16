# Ink

The ink stage converts sampled CT layer stacks into ink-probability maps with stock Reader v3.

## Modules

| Path | Purpose |
| --- | --- |
| `model/` | This module fetches Reader v3 and runs it with the Apple-silicon tile reader or the portable PyTorch reader. |
| `reading/` | This module samples each piece, selects its written face, and produces ink maps. |
| `datasets/` | This module exposes published labelled segments for the letter atlas. |

## Set up Reader v3

```sh
./run.sh ink.model.weights fetch
```

The command verifies the pinned Reader v3 download and stores it under `${HERCULEX_LIBRARY:-data/library}/weights/pretrained/reader_v3/`. HerculeX compiles the Neural Engine body on first use and stores the build under `weights/engine/` in the library.
