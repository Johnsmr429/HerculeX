# calibrate/survey

Reads bands of a scroll on the Neural Engine with a model not trained on it, forward against reversed, to judge
whether the scroll holds writing.

```text
scroll_survey.c        look, replicate, re-read and row samples of a band, forward against reversed
survey_fetch.c         a band's bricks fetched by their own job, ahead of the read
survey_rim_excess.c    sure ink excess with sheet rims cut away
survey_row_power.c     row power on forward and reversed maps

./run.sh calibrate.survey.scroll_survey look SCROLL/SCAN/BAND
./run.sh calibrate.survey.survey_fetch look|replicate|reread|row-sample|second SCROLL/SCAN/BAND
./run.sh runner.fetch_ahead apply      split waiting survey jobs into fetch and read
```
