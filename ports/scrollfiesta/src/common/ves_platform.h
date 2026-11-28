#ifndef VES_PLATFORM_H
#define VES_PLATFORM_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#if defined(_MSC_VER)
  #define VES_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
  #define VES_THREAD_LOCAL _Thread_local
#elif defined(__GNUC__)
  #define VES_THREAD_LOCAL __thread
#else
  #define VES_THREAD_LOCAL
#endif

#ifdef _MSC_VER
  #ifndef restrict
    #define restrict __restrict
  #endif

  #ifndef _CRT_SECURE_NO_WARNINGS
    #define _CRT_SECURE_NO_WARNINGS
  #endif

  #ifndef __attribute__
    #define __attribute__(x)
  #endif
#else
  #ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE 200809L
  #endif
#endif

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <direct.h>
  #include <io.h>
  #include <fcntl.h>
  #include <process.h>
  #include <signal.h>
  #include <time.h>
#else
  #include <unistd.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <time.h>
  #include <signal.h>
  #include <pthread.h>
  #include <dirent.h>
#endif

static inline double ves_clock_sec(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

#ifdef _WIN32

typedef SRWLOCK ves_mutex_t;

#define VES_MUTEX_INITIALIZER  SRWLOCK_INIT

static inline void ves_mutex_init(ves_mutex_t *m)
{
    InitializeSRWLock(m);
}

static inline void ves_mutex_lock(ves_mutex_t *m)
{
    AcquireSRWLockExclusive(m);
}

static inline void ves_mutex_unlock(ves_mutex_t *m)
{
    ReleaseSRWLockExclusive(m);
}

#else

typedef pthread_mutex_t ves_mutex_t;

#define VES_MUTEX_INITIALIZER  PTHREAD_MUTEX_INITIALIZER

static inline void ves_mutex_init(ves_mutex_t *m)
{
    pthread_mutex_init(m, NULL);
}

static inline void ves_mutex_lock(ves_mutex_t *m)
{
    pthread_mutex_lock(m);
}

static inline void ves_mutex_unlock(ves_mutex_t *m)
{
    pthread_mutex_unlock(m);
}

#endif

static inline int ves_cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 2;
#endif
}

static inline int ves_getpid(void)
{
#ifdef _WIN32
    return (int)_getpid();
#else
    return (int)getpid();
#endif
}

static inline int ves_unlink(const char *path)
{
#ifdef _WIN32
    return _unlink(path);
#else
    return unlink(path);
#endif
}

static inline int ves_mkdir(const char *path)
{
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

static inline int ves_rmdir(const char *path)
{
#ifdef _WIN32
    return _rmdir(path);
#else
    return rmdir(path);
#endif
}

#define VES_SELFTEST_DIR "selftest"

static inline int ves_selftest_dir(char *buf, size_t len, const char *name)
{
    int k = snprintf(buf, len, VES_SELFTEST_DIR "/%s_%d", name, ves_getpid());
    if (k < 0 || (size_t)k >= len) return -1;
    (void)ves_mkdir(VES_SELFTEST_DIR);
    (void)ves_mkdir(buf);
    return 0;
}

static inline int ves_remove_tree_(const char *path)
{
    char sub[1024];
    int rc = 0;
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES) return 0;
    if (attr & FILE_ATTRIBUTE_READONLY)
        (void)SetFileAttributesA(path, attr & ~(DWORD)FILE_ATTRIBUTE_READONLY);
    if (!(attr & FILE_ATTRIBUTE_DIRECTORY)) return DeleteFileA(path) ? 0 : -1;
    if (attr & FILE_ATTRIBUTE_REPARSE_POINT)
        return RemoveDirectoryA(path) ? 0 : -1;
    {
        WIN32_FIND_DATAA fd;
        HANDLE h;
        int k = snprintf(sub, sizeof sub, "%s/*", path);
        if (k < 0 || (size_t)k >= sizeof sub) return -1;
        h = FindFirstFileA(sub, &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (strcmp(fd.cFileName, ".") == 0 || strcmp(fd.cFileName, "..") == 0)
                    continue;
                k = snprintf(sub, sizeof sub, "%s/%s", path, fd.cFileName);
                if (k < 0 || (size_t)k >= sizeof sub || ves_remove_tree_(sub) != 0)
                    rc = -1;
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        }
    }
    if (!RemoveDirectoryA(path)) rc = -1;
#else
    struct stat st;
    DIR *d = NULL;
    if (lstat(path, &st) != 0) return 0;
    if (!S_ISDIR(st.st_mode)) return unlink(path) == 0 ? 0 : -1;
    d = opendir(path);
    if (d != NULL) {
        struct dirent *e = NULL;
        while ((e = readdir(d)) != NULL) {
            int k = 0;
            if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0)
                continue;
            k = snprintf(sub, sizeof sub, "%s/%s", path, e->d_name);
            if (k < 0 || (size_t)k >= sizeof sub || ves_remove_tree_(sub) != 0)
                rc = -1;
        }
        closedir(d);
    }
    if (rmdir(path) != 0) rc = -1;
#endif
    return rc;
}

static inline int ves_selftest_clean(const char *path)
{
    size_t n = strlen(VES_SELFTEST_DIR);
    if (path == NULL || strncmp(path, VES_SELFTEST_DIR, n) != 0
        || (path[n] != '/' && path[n] != '\\') || path[n + 1] == '\0'
        || strstr(path, "..") != NULL)
        return -1;
    return ves_remove_tree_(path);
}

static inline void ves_sleep_ms(int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

static inline int ves_temp_dir(char *buf, size_t len)
{
#ifdef _WIN32
    DWORD n = GetTempPathA((DWORD)len, buf);
    return (n > 0 && n < (DWORD)len) ? 0 : -1;
#else
    const char *tmp = "/tmp/";
    size_t slen = 5;
    if (slen >= len) return -1;
    memcpy(buf, tmp, slen + 1);
    return 0;
#endif
}

static inline const char *ves_path_basename(const char *path)
{
    const char *last_sep = NULL;
    const char *p = path;
    while (*p) {
        if (*p == '/' || *p == '\\')
            last_sep = p;
        p++;
    }
    return last_sep ? last_sep + 1 : path;
}

#if defined(_WIN32) || defined(__APPLE__)
  #define VES_FADVISE_DONTNEED  0
  static inline int ves_fadvise(int fd, long offset, long len, int advice)
  {
      (void)fd; (void)offset; (void)len; (void)advice;
      return 0;
  }
#else
  #include <fcntl.h>
  #define VES_FADVISE_DONTNEED  POSIX_FADV_DONTNEED
  static inline int ves_fadvise(int fd, long offset, long len, int advice)
  {
      return posix_fadvise(fd, offset, len, advice);
  }
#endif

#ifdef _WIN32
  static inline int ves_stdin_set_binary(void)
  {
      return (_setmode(_fileno(stdin), _O_BINARY) == -1) ? -1 : 0;
  }
#else
  static inline int ves_stdin_set_binary(void)
  {
      return 0;
  }
#endif

int ves_ensure_parent_dir(const char *filepath);

int ves_run_subprocess(const char *exe, const char *const *argv,
                       double timeout_sec);

int ves_run_subprocess_logged(const char *exe, const char *const *argv,
                              double timeout_sec, const char *log_path);

void ves_hard_timeout_start(double seconds, volatile sig_atomic_t *flag);
void ves_hard_timeout_cancel(void);

void ves_omp_set_threads(int n);
#if defined(_MSC_VER)
#define VES_NORETURN __declspec(noreturn)
#elif defined(__GNUC__)
#define VES_NORETURN __attribute__((noreturn))
#else
#define VES_NORETURN
#endif
VES_NORETURN void ves_fast_exit(int code);

#ifdef _WIN32
typedef unsigned (__stdcall *VesThreadFn)(void *);
#else
typedef unsigned (*VesThreadFn)(void *);
#endif
int ves_run_thread_cpu_bounded(VesThreadFn fn, void *arg, unsigned cpu_ms,
                               unsigned wall_ms, unsigned *out_cpu_ms);

int ves_platform_selftest(void);

void ves_prof_start_from_env(void);
void ves_prof_flush_now(void);

#endif
