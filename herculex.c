#include "base.h"
#include "storage/storage.h"
#include "scans/scans.h"
#include "sampling/sampling.h"
#include "level/detection/detection.h"
#include "level/level/level.h"
#include "level/layout/layout.h"
#include "ink/ink.h"
#if __has_include("synthetic/three_d/spec.c")
#define HX_LOCAL_SYNTHETIC 1
#include "synthetic/three_d/spec.c"
#include "synthetic/three_d/fields.c"
#include "synthetic/two_d/layout.c"
#include "synthetic/two_d/sampler.c"
#include "synthetic/two_d/model.c"
#include "synthetic/two_d/generate.c"
#include "synthetic/library.c"
#include "synthetic/injection/profile.c"
#ifdef __APPLE__
#include "synthetic/injection/papyrus.c"
#endif
#include "synthetic/two_d/discriminator.c"
#include "synthetic/three_d/roll.c"
#include "synthetic/three_d/deform.c"
#include "synthetic/three_d/families.c"
#include "synthetic/three_d/evaluate.c"
#include "synthetic/three_d/band.c"
#include "synthetic/three_d/sampler.c"
#include "synthetic/three_d/batch.c"
#include "synthetic/three_d/bench.c"
#include "synthetic/three_d/verify.c"
#include "synthetic/three_d/controls.c"
#include "synthetic/three_d/proof.c"
#include "synthetic/realism.c"
#ifdef __APPLE__
#include "synthetic/three_d/inject.c"
#endif
#include "synthetic/three_d/realism.c"
#ifdef __APPLE__
#include "synthetic/injection/fragment.c"
#include "synthetic/injection/fidelity.c"
#endif
#include "synthetic/three_d/tune.c"
#include "synthetic/three_d/solver_view.c"
#endif
#define HX_RULER_CLI
#include "calibrate/unwrap/raw_sheet_ruler.c"
#undef HX_RULER_CLI
#include "viewer/website.h"
#if defined(HX_LOCAL_SYNTHETIC) && defined(__APPLE__)
#include "synthetic/three_d/gallery.c"
#endif
#include "spots/spots.h"
#include "runner/runner.h"

typedef struct {
    const char *name;
    int (*run)(int, char **);
    const char *usage;
} Command;

int hx_route_index(int argc, char **argv);
int hx_check_main(int argc, char **argv);
int hx_scan_choice_main(int argc, char **argv);
int hx_scan_catalog_main(int argc, char **argv);
int hx_product_inventory_main(int argc, char **argv);
int hx_server_inventory_main(int argc, char **argv);
int hx_fetch_products_main(int argc, char **argv);
int hx_fetch_server_main(int argc, char **argv);
int hx_chunk_usage_main(int argc, char **argv);
int hx_chunk_evict_main(int argc, char **argv);
int hx_chunk_resolve_main(int argc, char **argv);
int hx_chunk_fetch_main(int argc, char **argv);
int hx_wrap_study_main(int argc, char **argv);
int hx_wrap_survey_main(int argc, char **argv);
int hx_wrap_truth_main(int argc, char **argv);
int hx_mesh_reference_main(int argc, char **argv);
int hx_sheet_scoring_main(int argc, char **argv);
int hx_controls_main(int argc, char **argv);

int catalog_command(int argc, char **argv) { return hx_scan_catalog_main(argc + 1, argv - 1); }

int paths_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("Usage: ./run.sh paths [data|library|outputs]");
        return 0;
    }
    need(argc <= 1, "usage: ./run.sh paths [data|library|outputs]");
    const char *names[] = {"data", "library", "outputs"};
    const char *paths[] = {DATA, LIBRARY, OUTPUTS};
    for (int i = 0; i < 3; i++) {
        if (!argc) {
            printf("%s\t%s\n", names[i], paths[i]);
        } else if (!strcmp(argv[0], names[i])) {
            puts(paths[i]);
            return 0;
        }
    }
    need(!argc, "unknown data root: %s", argv[0]);
    return 0;
}

int bench_package_command(int argc, char **argv) {
    if (argc && !strcmp(argv[0], "build")) {
        return bench_package_build_command(argc - 1, argv + 1);
    }
    return bench_package_inspect_command(argc, argv);
}

int check_command(int argc, char **argv) {
    if (argc && !strcmp(argv[0], "lint")) {
        return lint_command(argc - 1, argv + 1);
    }
    return hx_check_main(argc + 1, argv - 1);
}

#if defined(HX_LOCAL_SYNTHETIC) && defined(__APPLE__)
static int syn_inject_dispatch(int argc, char **argv) {
    if (argc && !strcmp(argv[0], "render")) {
        return syn_inject_command(argc - 1, argv + 1);
    }
    if (argc && !strcmp(argv[0], "realism")) {
        return syn_injected_realism_command(argc - 1, argv + 1);
    }
    int help = argc && (!strcmp(argv[0], "-h") || !strcmp(argv[0], "--help"));
    fputs("Usage: ./run.sh synthetic.three_d.inject {render|realism} [ARGS]\n",
        help ? stdout : stderr);
    return help ? 0 : 2;
}
#endif

Command commands[] = {
    {"level", level_command, "SCROLL[:SCAN] ...  level 1.1 CT to published page"},
    {"status", hx_pipeline_status, "[SCROLL ...] [--scan ID]  what is done"},
    {"catalog", catalog_command, "[SCROLL]  the scan catalog and each scroll's survey scan"},
    {"paths", paths_command, "[data|library|outputs]  resolved data folders"},
    {"sheets", sheets_command, "SCROLL [--scan ID]  sheet nodes per unit of brick rows"},
    {"pieces", pieces_command, "SCROLL [--scan ID]  winding levels, pieces and the frame"},
    {"fit", fit_command, "SCROLL SCAN UNIT  the level fit of one unit"},
    {"curl", curl_command, "SCROLL SCAN 1|-1  set a scan's curl by hand"},
    {"umbilicus", umbilicus_command, "SCROLL SCAN [SOURCE]  show or keep a scan's umbilicus"},
    {"layout", layout_command, "SCROLL [--scan ID]  every piece placed on one sheet"},
    {"sample", sample_command, "SCROLL SCAN UNIT PIECE  one piece's layer stack"},
    {"fetch", fetch_command, "SCROLL [--scan ID]  CT for the read, fetched as units finish"},
    {"ink", ink_command, "SCROLL SCAN  layer stacks, written faces and ink maps"},
    {"models", models_command, "[add NAME TITLE FILE [reader|studio]]  the ink models"},
    {"weights", weights_command, "fetch [MODEL]|import MODEL FILE|status  ink checkpoints"},
    {"ink.model.models", models_command, ""},
    {"ink.model.weights", weights_command, ""},
    {"publish", publish_command, "SCROLL SCAN  tiles, viewer page and start-page card"},
    {"serve", serve_command, "[PORT]  the local viewer and its mark store"},
    {"card", card_command, "PAGE  put a page on the start page"},
    {"spots", spots_command, "[TABLE|verdict ...]  circled spots"},
    {"board", board_command, "[COMMAND]  queue and monitor jobs"},
    {"check", check_command, "[cpu|all|lint]  golden and end-to-end checks"},
    {"timeline", timeline_command, "[JOB ...]  where jobs spent their time"},
    {"disk", disk_command, "[report|reclaimable] [--depth N]  data storage usage"},
    {"port", port_command, "PORT TARGET  build an optional native tool"},
    {"runner.job_board", board_command, ""},
    {"runner.timeline", timeline_command, ""},
    {"runner.disk", disk_command, ""},
#ifdef HX_LOCAL_SURVEY
    {"runner.fetch_ahead", fetch_ahead_command, ""},
#endif
    {"runner.memory_trace", memory_trace_command, ""},
    {"runner.place_build", place_build_command, ""},
    {"runner.speed", speed_command, ""},
    {"runner.native_ports", port_command, ""},
    {"runner.check", check_command, ""},
    {"runner.lint", lint_command, ""},
    {"runner.keep_faces", keep_faces_command, ""},
#ifdef HX_LOCAL_DATA_AUDIT
    {"data.catalog.catalog", data_audit_command, ""},
#endif
    {"level.pipeline", hx_pipeline_main, 0},
    {"runner.sheet_stage", hx_sheet_main, 0},
    {"level.unwrap", hx_unwrap_main, 0},
    {"runner.review_page", publish_command, ""},
    {"storage.route_database", hx_route_index, 0},
    {"scans.catalog.scan_catalog", hx_scan_catalog_main, 0},
    {"scans.catalog.scan_choice", hx_scan_choice_main, 0},
    {"scans.catalog.product_inventory", hx_product_inventory_main, 0},
    {"scans.catalog.server_inventory", hx_server_inventory_main, 0},
    {"scans.transfer.fetch_products", hx_fetch_products_main, 0},
    {"scans.transfer.fetch_server", hx_fetch_server_main, 0},
    {"scans.chunks.usage", hx_chunk_usage_main, 0},
    {"scans.chunks.evict", hx_chunk_evict_main, 0},
    {"scans.chunks.resolve", hx_chunk_resolve_main, 0},
    {"scans.chunks.fetch", hx_chunk_fetch_main, 0},
    {"level.detection.detect_nodes", hx_detection_main, 0},
    {"level.level.run_levels", hx_levels_main, 0},
    {"level.level.umbilicus", hx_umbilicus_main, 0},
    {"level.layout.seams", hx_seams_main, 0},
    {"level.layout.mosaic_layout", hx_layout_rows_main, 0},
    {"level.level.scroll_layout", hx_layout_framed_main, 0},
    {"runner.pipeline", hx_pipeline_main, 0},
    {"runner.unwrap", hx_unwrap_main, 0},
    {"unwrapping.detection.detect_nodes", hx_detection_main, 0},
    {"unwrapping.level.run_levels", hx_levels_main, 0},
    {"unwrapping.level.umbilicus", hx_umbilicus_main, 0},
    {"unwrapping.layout.seams", hx_seams_main, 0},
    {"unwrapping.layout.mosaic_layout", hx_layout_rows_main, 0},
    {"unwrapping.level.scroll_layout", hx_layout_framed_main, 0},
    {"viewer.sheet_tiles", hx_sheet_tiles_main, 0},
    {"viewer.refresh_viewers", hx_refresh_viewers_main, 0},
    {"viewer.site_server", serve_command, ""},
    {"viewer.survey_status", hx_survey_status_command, ""},
    {"viewer.site_pages", hx_site_pages_command, ""},
    {"viewer.panel_pages", hx_panel_pages_command, ""},
    {"viewer.overview", hx_overview_command, ""},
    {"viewer.window_proposals", hx_window_proposals_command, ""},
    {"viewer.mosaic_outline", hx_mosaic_outline_command, ""},
    {"viewer.atlas_build", hx_atlas_command, ""},
    {"website.sheet_tiles", hx_sheet_tiles_main, 0},
    {"website.refresh_viewers", hx_refresh_viewers_main, 0},
    {"website.site_server", serve_command, ""},
    {"website.survey_status", hx_survey_status_command, ""},
    {"website.site_pages", hx_site_pages_command, ""},
    {"website.panel_pages", hx_panel_pages_command, ""},
    {"website.overview", hx_overview_command, ""},
    {"website.window_proposals", hx_window_proposals_command, ""},
    {"website.mosaic_outline", hx_mosaic_outline_command, ""},
    {"website.atlas_build", hx_atlas_command, ""},
    {"spots.database", spots_command, ""},
    {"spots.catalogue", hx_spot_catalogue, ""},
    {"spots.pilot", spot_pilot_command, ""},
    {"spots.reflatten", spot_reflatten_command, ""},
    {"spots.flatten", spot_flatten_command, ""},
#ifdef HX_LOCAL_SURVEY
    {"calibrate.survey.survey_rim_excess", survey_rim_command, ""},
    {"calibrate.survey.survey_row_power", survey_row_power_command, ""},
    {"calibrate.survey.scroll_survey", survey_command, ""},
    {"calibrate.survey.survey_fetch", survey_fetch_command, ""},
    {"calibration.survey.survey_rim_excess", survey_rim_command, ""},
    {"calibration.survey.survey_row_power", survey_row_power_command, ""},
    {"calibration.survey.scroll_survey", survey_command, ""},
    {"calibration.survey.survey_fetch", survey_fetch_command, ""},
#endif
    {"calibrate.unwrap.sheet_wrap_study", hx_wrap_study_main, 0},
    {"calibrate.unwrap.sheet_wrap_survey", hx_wrap_survey_main, 0},
    {"calibrate.unwrap.sheet_wrap_truth", hx_wrap_truth_main, 0},
    {"calibrate.unwrap.mesh_reference", hx_mesh_reference_main, 0},
    {"calibrate.unwrap.sheet_scoring", hx_sheet_scoring_main, 0},
    {"calibrate.unwrap.raw_sheet_ruler", hx_raw_ruler_main, 0},
    {"calibrate.unwrap.sheet_controls", hx_controls_main, 0},
    {"calibrate.bench.score", bench_score_command, ""},
    {"calibrate.bench.score_check", bench_score_check_command, ""},
    {"calibrate.bench.results", bench_results_command, ""},
    {"calibrate.bench.package", bench_package_command, ""},
    {"calibrate.bench.truth", bench_truth_command, ""},
    {"calibrate.bench.surface_score", bench_surface_score_command, ""},
    {"calibrate.bench.winding_score", bench_winding_score_command, ""},
    {"calibrate.bench.ridge", bench_ridge_command, ""},
    {"calibrate.bench.qa", bench_qa_command, ""},
    {"calibrate.bench.v1_sources", bench_v1_sources_command, ""},
    {"calibration.unwrap.sheet_wrap_study", hx_wrap_study_main, 0},
    {"calibration.unwrap.sheet_wrap_survey", hx_wrap_survey_main, 0},
    {"calibration.unwrap.sheet_wrap_truth", hx_wrap_truth_main, 0},
    {"calibration.unwrap.mesh_reference", hx_mesh_reference_main, 0},
    {"calibration.unwrap.sheet_scoring", hx_sheet_scoring_main, 0},
    {"calibration.unwrap.raw_sheet_ruler", hx_raw_ruler_main, 0},
    {"calibration.unwrap.sheet_controls", hx_controls_main, 0},
    {"calibration.bench.score", bench_score_command, ""},
    {"calibration.bench.score_check", bench_score_check_command, ""},
    {"calibration.bench.results", bench_results_command, ""},
    {"calibration.bench.package", bench_package_command, ""},
    {"calibration.bench.truth", bench_truth_command, ""},
    {"calibration.bench.surface_score", bench_surface_score_command, ""},
    {"calibration.bench.winding_score", bench_winding_score_command, ""},
    {"calibration.bench.ridge", bench_ridge_command, ""},
    {"calibration.bench.qa", bench_qa_command, ""},
    {"calibration.bench.v1_sources", bench_v1_sources_command, ""},
#ifdef HX_LOCAL_SYNTHETIC
    {"synthetic.three_d.spec", syn_spec_command, ""},
    {"synthetic.three_d.families", syn_families_command, ""},
    {"synthetic.three_d.band", syn_band_command, ""},
    {"synthetic.three_d.batch", syn_batch_command, ""},
    {"synthetic.three_d.sampler", syn_sampler_command, ""},
    {"synthetic.three_d.bench", syn_bench_command, ""},
    {"synthetic.three_d.verify", syn_verify_command, ""},
    {"synthetic.three_d.controls", syn_controls_command, ""},
    {"synthetic.three_d.proof", syn_proof_command, ""},
    {"synthetic.three_d.realism", syn_realism_command, ""},
    {"synthetic.three_d.tune", syn_tune_command, ""},
    {"synthetic.three_d.discriminator", syn_discriminator_command, ""},
    {"synthetic.three_d.solver_view", syn_solver_view_command, ""},
    {"synthetic.realism", realism_command, ""},
    {"synthetic.library", slab_library_command, ""},
    {"synthetic.library.native", slab_library_command, ""},
    {"synthetic.injection.profile", injection_profile_command, ""},
    {"synthetic.injection.profile.native", injection_profile_command, ""},
    {"synthetic.two_d.layout", slices_layout_command, ""},
    {"synthetic.two_d.generate.cpu", slices_generate_command, ""},
    {"synthetic.two_d.generate", slices_generate_native_command, ""},
    {"synthetic.two_d.generate.native", slices_generate_native_command, ""},
    {"synthetic.generate", slices_generate_native_command, ""},
    {"synthetic.two_d.discriminator", slices_discriminator_command, ""},
    {"synthetic.discriminator", slices_discriminator_command, ""},
#ifdef __APPLE__
    {"synthetic.injection.papyrus", injection_papyrus_command, ""},
    {"synthetic.injection.fragment", injection_fragment_command, ""},
    {"synthetic.injection.fragment.native", injection_fragment_command, ""},
    {"synthetic.injection.fidelity", injection_fidelity_command, ""},
    {"synthetic.injection.fidelity.native", injection_fidelity_command, ""},
    {"synthetic.three_d.gallery", syn_gallery_command, ""},
    {"synthetic.three_d.inject", syn_inject_dispatch, ""},
    {"synthetic.three_d.inject.render", syn_inject_command, ""},
    {"synthetic.three_d.inject.realism", syn_injected_realism_command, ""},
#endif
#endif
#ifdef HX_LOCAL_TEXT
    {"calibrate.text.line_spectrum", line_spectrum_command, ""},
    {"calibrate.text.readability_judge", readability_judge_command, ""},
    {"calibration.text.line_spectrum", line_spectrum_command, ""},
    {"calibration.text.readability_judge", readability_judge_command, ""},
#endif
};

int main(int argc, char **argv) {
    roots_init();
    int n = sizeof(commands) / sizeof(*commands);
    for (int i = 0; argc > 1 && i < n; i++) {
        if (!strcmp(argv[1], commands[i].name)) {
            return commands[i].usage ? commands[i].run(argc - 2, argv + 2)
                                     : commands[i].run(argc - 1, argv + 1);
        }
    }
    if (argc > 1 && !strcmp(argv[1], "has")) {
        for (int i = 0; argc > 2 && i < n; i++) {
            if (!strcmp(argv[2], commands[i].name)) {
                return 0;
            }
        }
        return 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--list-native")) {
        for (int i = 0; i < n; i++) {
            printf("%s\t%s\n", commands[i].name, commands[i].usage ? commands[i].usage : "");
        }
        return 0;
    }
    if (argc > 1) {
        fprintf(stderr, "Unknown native command %s. Run ./run.sh list.\n", argv[1]);
        return 2;
    }
    for (int i = 0; i < n; i++) {
        if (commands[i].usage && *commands[i].usage) {
            printf("%-10s %s\n", commands[i].name, commands[i].usage);
        }
    }
    return 0;
}
