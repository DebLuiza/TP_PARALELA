#ifndef KMEANS_WINDOWS_TIME_COMPAT_H
#define KMEANS_WINDOWS_TIME_COMPAT_H

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <time.h>

static inline void kmeans_monotonic_time(struct timespec *ts) {
    LARGE_INTEGER ticks;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);

    ts->tv_sec = (time_t)(ticks.QuadPart / frequency.QuadPart);
    ts->tv_nsec = (long)((ticks.QuadPart % frequency.QuadPart) * 1e9
                        / frequency.QuadPart);
}

#define CLOCK_MONOTONIC 0
#define clock_gettime(clock_id, ts) kmeans_monotonic_time(ts)

#endif
