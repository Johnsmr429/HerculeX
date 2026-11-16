# ink/training

```text
model_training.py             the recipe: stem and body taught on labelled cases, published pairs, far faces and scrambled target papyrus as blank
left_out_training_runs.py     the recipe with each labelled scroll left out, scored on it
reader_v3_start_training.py   the recipe begun from Reader v3 weights
map_context_reader.py         a second reader that learns the published map from the first reader's map
depth_mean_distillation.py    Reader v3 taught to give from one window its own mean over depths -4..+4
letter_classifier.py          three small networks that name a Greek letter from a window of ink map, taught on datasets/eye_letters

./run.sh ink.training.model_training NAME TARGET [LEFT_OUT] [SEED] [STEPS]
./run.sh ink.training.left_out_training_runs STEPS
./run.sh ink.training.letter_classifier holdout | scrolls | final     minutes, not hours; final writes weights/trained/letter-classifier.pth
```

Reads `datasets/` (schema in `../datasets/README.md`), writes `weights/trained/NAME.pth` with `NAME.json`.
Each run is scored on the labelled cases of the scroll left out of its training.
Training holds the GPU for hours: add it to the board with `--gpu`.
