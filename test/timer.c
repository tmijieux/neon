#include <string.h>
#include <stdio.h>

#include "timer.h"
#include "util.h"

void timer_init(struct timer *timer)
{
    memset(timer, 0, sizeof *timer);
    timer->running = false;
    timer->length = 0.0;
}

void timer_reset(struct timer *timer)
{
    timer_init(timer);
}

void timer_start(struct timer *timer)
{
    if (timer->running) {
        NEON_FATAL_ERROR("this timer is already started.");
    }
    gettimeofday(&timer->last_start, NULL);
    timer->running = true;
}

#define TIMEVAL_DIFF(a, b) (  ((a).tv_sec - (b).tv_sec) + ( ((a).tv_usec - (b).tv_usec) / 1000000.0)  )

void timer_stop(struct timer *timer)
{
    if (!timer->running) {
        NEON_FATAL_ERROR("this timer is not running.");
    }
    struct timeval tsp;
    gettimeofday(&tsp, NULL);
    timer->length += TIMEVAL_DIFF(tsp, timer->last_start);
    timer->running = false;
}

double timer_get_length(struct timer *timer)
{
    if (timer->running) {
        struct timeval tsp;
        gettimeofday(&tsp, NULL);
        double length = timer->length + TIMEVAL_DIFF(tsp, timer->last_start);
        return length;
    }
    return timer->length;
}

double timer_get_micro_length(struct timer *timer)
{
    if (timer->running) {
        struct timeval tsp;
        gettimeofday(&tsp, NULL);
        double length = timer->length + TIMEVAL_DIFF(tsp, timer->last_start);
        return length * 1000000.0;
    }
    return timer->length * 1000000.0;
}

