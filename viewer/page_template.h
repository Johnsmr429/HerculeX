#pragma once

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char viewer_template[] = "viewer/assets/sheet_viewer.html";

static char *read_page_text(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        return NULL;
    }
    char *text = NULL;
    struct stat info;
    if (!fstat(fileno(file), &info) && S_ISREG(info.st_mode) && info.st_size >= 0 &&
        info.st_size < INT_MAX) {
        size_t size = info.st_size;
        text = malloc(size + 1);
        if (text && fread(text, 1, size, file) == size) {
            text[size] = 0;
        } else {
            free(text);
            text = NULL;
        }
    }
    fclose(file);
    return text;
}

static int render_viewer(FILE *out, const char *page, const char *const values[4]) {
    const char *fields[] = {"__TITLE__", "__META__", "__FLOOR__", "__SCAN__"};
    while (*page) {
        const char *token = strstr(page, "__");
        size_t length = token ? (size_t)(token - page) : strlen(page);
        if (fwrite(page, 1, length, out) != length) {
            return 0;
        }
        if (!token) {
            return 1;
        }
        size_t found = 0;
        while (found < 4 && strncmp(token, fields[found], strlen(fields[found])) != 0) {
            found++;
        }
        if (found < 4) {
            if (fputs(values[found], out) < 0) {
                return 0;
            }
            page = token + strlen(fields[found]);
        } else {
            if (fputs("__", out) < 0) {
                return 0;
            }
            page = token + 2;
        }
    }
    return 1;
}
