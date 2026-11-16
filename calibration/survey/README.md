# calibration/survey

Reads every scroll on the Neural Engine with a model not trained on it, and ranks the scrolls for the prizes.

```text
scroll_survey.py        look, replicate, re-read and row samples of a band, forward against reversed
survey_fetch.py         a band's bricks fetched by their own job, ahead of the read
survey_rim_excess.py    sure ink excess with sheet rims cut away
survey_row_power.py     row power on forward and reversed maps
scroll_leaderboard.py   prize scrolls ranked by a protocol fixed before the reads

./run.sh calibration.survey.scroll_survey look SCROLL/SCAN/BAND
./run.sh calibration.survey.survey_fetch look|replicate|second SCROLL/SCAN/BAND
./run.sh runner.fetch_ahead apply      split waiting survey jobs into fetch and read
```
