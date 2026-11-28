int hx_unwrap_main(int argc, char **argv);
int hx_pipeline_main(int argc, char **argv);
int hx_sheet_main(int argc, char **argv);
int hx_pipeline_status(int argc, char **argv);

int level_command(int argc, char **argv) { return hx_unwrap_main(argc + 1, argv - 1); }

#include "native_ports.c"
#include "lint.c"
#include "lane.c"
#include "job_board.c"
#include "events.c"
#include "stage_queue.c"
#include "keep_faces.c"
#include "timeline.c"
#include "disk.c"
#include "place_build.c"
#include "memory_trace.c"
#if __has_include("../calibrate/survey/scroll_survey.c")
#define HX_LOCAL_SURVEY 1
#include "fetch_ahead.c"
#include "../calibrate/survey/scroll_survey.c"
#include "../calibrate/survey/survey_rim_excess.c"
#include "../calibrate/survey/survey_row_power.c"
#include "../calibrate/survey/survey_fetch.c"
#endif
#if __has_include("../data/catalog/catalog.c")
#define HX_LOCAL_DATA_AUDIT 1
#include "../data/catalog/catalog.c"
#endif
#include "speed.c"

#if __has_include("../calibrate/text/line_spectrum.c")
#define HX_LOCAL_TEXT 1
#include "../calibrate/text/line_spectrum.c"
#include "../calibrate/text/readability_judge.c"
#endif
#include "../calibrate/bench/score.c"
#include "../calibrate/bench/v1_sources.c"
#include "../calibrate/bench/score_check.c"
#include "../calibrate/bench/results.c"
#include "../calibrate/bench/package.c"
#include "../calibrate/bench/truth.c"
#include "../calibrate/bench/surface_score.c"
#include "../calibrate/bench/winding_score.c"
#include "../calibrate/bench/ridge.c"
#include "../calibrate/bench/qa.c"
