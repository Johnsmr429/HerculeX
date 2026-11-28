#include "../scans/chunks/cached_ct.c"

#define VERIFY(expression)                                                                         \
    do {                                                                                           \
        if (!(expression)) {                                                                       \
            fprintf(stderr, "CT chunks golden failed at line %d: %s\n", __LINE__, #expression);    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static void fixture(const char *path, const char *sql) {
    sqlite3 *db = NULL;
    VERIFY(sqlite3_open(path, &db) == SQLITE_OK);
    VERIFY(sqlite3_exec(db, sql, NULL, NULL, NULL) == SQLITE_OK);
    VERIFY(sqlite3_close(db) == SQLITE_OK);
}

int main(void) {
    char temp[] = "/tmp/hx-cache % ?-XXXXXX";
    const char *root = mkdtemp(temp);
    VERIFY(root);
    char sample[128], folder[128], raw[128], registry[128], index[128];
    snprintf(sample, sizeof(sample), "%s/S", root);
    snprintf(folder, sizeof(folder), "%s/S/array/0", root);
    snprintf(raw, sizeof(raw), "%s/S/array/0/chunks.pack", root);
    snprintf(registry, sizeof(registry), "%s/arrays.sqlite", root);
    snprintf(index, sizeof(index), "%s/S/array/0/index.sqlite", root);
    VERIFY(ct_make_folder(folder));
    uint8_t pack[48];
    for (int i = 0; i < 48; i++) {
        pack[i] = i;
    }
    int fd = open(raw, O_WRONLY | O_CREAT, 0600);
    VERIFY(fd >= 0 && write(fd, pack, sizeof(pack)) == sizeof(pack) && close(fd) == 0);
    fixture(registry,
        "PRAGMA journal_mode=WAL;"
        "CREATE TABLE resolved(sample,scan,kind,url);"
        "CREATE TABLE arrays(sample,name,shape,chunks,fill_value,dtype,level,root,url);"
        "INSERT INTO resolved VALUES('S','test','ct','fixture/');"
        "INSERT INTO arrays VALUES('S','array','[3,5,7]','[2,3,4]',17,'uint8',0,"
        "'fixture','fixture/0');");
    fixture(index,
        "PRAGMA journal_mode=WAL;"
        "CREATE TABLE chunks(z,y,x,offset,PRIMARY KEY(z,y,x)) WITHOUT ROWID;"
        "INSERT INTO chunks VALUES(0,0,0,0),(0,0,1,-1),(1,1,1,24);");
    VERIFY(chmod(root, 0555) == 0 && chmod(folder, 0555) == 0);
    HxCachedCt ct;
    VERIFY(hx_cached_ct_open(&ct, root, "S", "test", 0) == 0);
    const int64_t box[] = {0, 4, 0, 6, 0, 8};
    uint8_t result[4][6][8];
    int64_t unknown;
    VERIFY(hx_cached_ct_read(&ct, box, &result[0][0][0], &unknown) == 0);
    VERIFY(unknown == 5);
    for (int z = 0; z < 4; z++) {
        for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 8; x++) {
                int expected = 17;
                if (z < 2 && y < 3 && x < 4) {
                    expected = (z * 3 + y) * 4 + x;
                } else if (z == 2 && y >= 3 && y < 5 && x >= 4 && x < 7) {
                    expected = 24 + (y - 3) * 4 + x - 4;
                }
                VERIFY(result[z][y][x] == expected);
            }
        }
    }
    hx_cached_ct_close(&ct);
    VERIFY(chmod(root, 0755) == 0 && chmod(folder, 0755) == 0);
    const char *files[] = {index, registry};
    const char *sidecars[] = {"-shm", "-wal"};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            char *path = NULL;
            VERIFY(asprintf(&path, "%s%s", files[i], sidecars[j]) >= 0);
            VERIFY(unlink(path) == 0 || errno == ENOENT);
            free(path);
        }
    }
    VERIFY(unlink(raw) == 0 && unlink(index) == 0 && unlink(registry) == 0);
    VERIFY(rmdir(folder) == 0);
    *strrchr(folder, '/') = 0;
    VERIFY(rmdir(folder) == 0 && rmdir(sample) == 0 && rmdir(root) == 0);
    puts("CT chunks golden passed");
    return 0;
}
