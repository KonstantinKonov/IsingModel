#ifndef __STAT_H__
#define __STAT_H__

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

void flatten_double_2D(double **arr, int n, double **res);
void flatten_point_arr_to_double(double ***arr, int n, int m, double **res);
void linspace(double min, double max, int n, double **res);

#endif