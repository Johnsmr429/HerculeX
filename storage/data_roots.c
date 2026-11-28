#pragma once
typedef struct {
    const char *folder, *name, *older;
} RootSetting;

static const RootSetting library_root = {"library", "HERCULEX_LIBRARY", "HERCULEX_DISC_PCIE"};
static const RootSetting outputs_root = {"outputs", "HERCULEX_OUTPUTS", "HERCULEX_DISC_PCLE3"};

typedef struct {
    const char *name, *parent, *relative;
} PathEntry;

static const PathEntry locations[] = {
    {"RUNTIME", "HERE", "runtime"},
    {"SCRATCH", "RUNTIME", "scratch"},
    {"KERNELS", "RUNTIME", "kernels"},
    {"BROWSER_ASSETS", "HERE", "viewer/assets"},
    {"RECORDS", "HERE", "records"},
    {"INK_RECORDS", "RECORDS", "ink"},
    {"SURFACE_RECORDS", "RECORDS", "surface"},
    {"SCAN_CATALOG", "LIBRARY", "catalog"},
    {"RAW_SCANS", "LIBRARY", "raw"},
    {"OPEN_DATA_MIRROR", "LIBRARY", "published/open_data"},
    {"DL_ASH2TXT_MIRROR", "LIBRARY", "published/dl_ash2txt"},
    {"INK_TRAINING_DATA", "LIBRARY", "datasets"},
    {"LABELLED_CASES", "INK_TRAINING_DATA", "cases"},
    {"INFRARED_FRAGMENTS", "INK_TRAINING_DATA", "fragments"},
    {"HAND_MARKS", "LIBRARY", "marks"},
    {"OWN_SURFACE_HOLDOUT", "INK_TRAINING_DATA", "holdout"},
    {"MODEL_WEIGHTS", "LIBRARY", "weights"},
    {"INK_WEIGHTS", "MODEL_WEIGHTS", "ink"},
    {"SURFACE_WEIGHTS", "MODEL_WEIGHTS", "surface"},
    {"NEURAL_ENGINE_BUILDS", "MODEL_WEIGHTS", "engine"},
    {"EXTERNAL_TOOLS", "LIBRARY", "tools"},
    {"SHEET_CONTROLS", "LIBRARY", "controls/sheets"},
    {"REFERENCE_SURFACES", "LIBRARY", "controls/surfaces"},
    {"SURFACE_PREDICTIONS", "OUTPUTS", "unwrapping/surface"},
    {"CALIBRATION_OUTPUTS", "LIBRARY", "calibration"},
    {"STUDIES", "LIBRARY", "studies"},
    {"SPOT_CANDIDATES", "STUDIES", "shared/spot-candidates"},
    {"SUBMISSIONS", "LIBRARY", "submissions"},
    {"UNWRAP_OUTPUTS", "OUTPUTS", "unwrapping"},
    {"NODE_ROUTE", "UNWRAP_OUTPUTS", "nodes"},
    {"DETECTION_WORK", "UNWRAP_OUTPUTS", "detection-work"},
    {"LEVEL_ROUTE", "UNWRAP_OUTPUTS", "level"},
    {"ROUTE_OUTPUTS", "LEVEL_ROUTE", ""},
    {"ML_OUTPUTS", "UNWRAP_OUTPUTS", "network"},
    {"NETWORK_TRAINING", "UNWRAP_OUTPUTS", "network-training"},
    {"WINDING_LABELS", "NETWORK_TRAINING", "labels"},
    {"WINDING_MODELS", "NETWORK_TRAINING", "models"},
    {"SURFACE_BENCH", "OUTPUTS", "bench"},
    {"OPEN_DATA_EXTRA", "OUTPUTS", "cache/open_data"},
    {"INK_LAYER_STACKS", "OUTPUTS", "stacks"},
    {"WEBSITE", "OUTPUTS", "website"},
    {"SHEETS", "WEBSITE", "sheets"},
    {"CIRCLE_REVIEW", "SHEETS", "circles-PHerc0211/review"},
    {"SITE_ATLAS", "WEBSITE", "atlas"},
};

#include <errno.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *path_join(const char *a, const char *b) {
    char *result = NULL;
    if (asprintf(&result, "%s/%s", a, b) < 0)
        return NULL;
    return result;
}

static char *expanded(const char *path) {
    if (path[0] != '~')
        return strdup(path);
    const char *slash = strchr(path, '/');
    size_t n = slash ? (size_t)(slash - path - 1) : strlen(path + 1);
    char *user = strndup(path + 1, n);
    if (!user)
        return NULL;
    const char *home = NULL;
    if (!n)
        home = getenv("HOME");
    if (!home) {
        struct passwd *pw = n ? getpwnam(user) : getpwuid(getuid());
        if (pw)
            home = pw->pw_dir;
    }
    free(user);
    if (!home) {
        errno = ENOENT;
        return NULL;
    }
    return slash ? path_join(home, slash + 1) : strdup(home);
}

static char *resolved(const char *path) {
    char *name = expanded(path);
    if (!name) {
        return NULL;
    }
    char *absolute = realpath(name, NULL);
    if (!absolute && name[0] != '/') {
        char *cwd = getcwd(NULL, 0);
        absolute = cwd ? path_join(cwd, name) : NULL;
        free(cwd);
    }
    if (!absolute) {
        return name;
    }
    free(name);
    return absolute;
}

static char *data_root(void) {
    const char *data = getenv("HERCULEX_DATA");
    return resolved(data && *data ? data : "data");
}

static char *local_root(RootSetting setting) {
    const char *path = getenv(setting.name);
    if (!path || !*path)
        path = getenv(setting.older);
    if (path && *path)
        return resolved(path);
    char *base = data_root();
    char *name = base ? path_join(base, setting.folder) : NULL;
    char *result = name ? resolved(name) : NULL;
    free(name);
    free(base);
    return result;
}
