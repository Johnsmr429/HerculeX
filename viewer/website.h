#pragma once

#include <setjmp.h>
#include "page_template.h"
#include "sheet_page.h"

int hx_sheet_tiles_main(int argc, char **argv);
int hx_refresh_viewers_main(int argc, char **argv);
int hx_refresh_title(const char *path, const char *page, const char *title);
int hx_check_review(int argc, char **argv);

#include "site_server.c"
#include "survey_status.c"
#include <glob.h>
#include "suggested_areas.c"
#include "window_proposals.c"
#include "site_pages.c"

#include "panel_pages.c"
#include "overview.c"
#include "mosaic_outline.c"
#include "atlas_build.c"
