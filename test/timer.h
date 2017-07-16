#ifndef NEON_TIMER_H
#define NEON_TIMER_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>

/**
 * \brief time measurements facilities (chronometer)
 */
struct timer
{
    bool running;
    struct timeval last_start;
    double length;
};

void timer_init(struct timer *timer);
void timer_reset(struct timer *timer);
void timer_start(struct timer *timer);
void timer_stop(struct timer *timer);
double timer_get_length(struct timer *timer);
double timer_get_micro_length(struct timer *timer);

#endif // NEON_TIMER_H
