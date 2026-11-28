# calibrate/text

Tests whether the ink maps show writing.

```text
line_spectrum.c        power at a line spacing across the fibres; a model that sees 1.2 mm cannot invent rows 5 mm apart
readability_judge.c     compares two maps shown blind to a vision model

./run.sh calibrate.text.line_spectrum IMAGE.npy MASK.npy OUT.json [SETTINGS.json]
./run.sh calibrate.text.readability_judge PAGE FIRST_TITLE SECOND_TITLE [SEED]
```
