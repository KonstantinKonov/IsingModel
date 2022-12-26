#ifndef __RAND_H__
#define __RAND_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void rand_init();
int rand_int(int min, int max);
double rand_double(double min, double max);
double draw_new_x(double x, double step_size, double border_low, double border_high);

#endif