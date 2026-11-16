# calibration/text

Tests whether the ink maps show writing.

```text
line_spectrum.py   power at a line spacing across the fibres; a model that sees 1.2 mm cannot invent rows 5 mm apart
blind_pairs.py     one square read from each face, unlabelled, in a sealed order, for judging by eye; images in website/blind_pairs/
spot_tests.py      strokes at one spot tested across seeds, grids, depths, the other face and a structureless twin
readability_judge.py   two maps of a passage shown blind to a vision model: which reads clearer, which letters it names

./run.sh calibration.text.spot_tests SCROLL SCAN SPOTS.json
./run.sh calibration.text.blind_pairs fetch | read | score ANSWERS
./run.sh calibration.text.readability_judge PAGE_NAME FIRST_TITLE SECOND_TITLE
```
