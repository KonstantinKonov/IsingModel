#ifndef __EXPORT_H__
#define __EXPORT_H__

#include <stdio.h>
#include <stdlib.h>
#include "stat.h"
#include "rast.h"
#include "types.h"

void export_double_2D(double **arr, int n, char *filename);
void export_double_1D(double *arr, int n, char *filename);
void export_point_2D(double ***arr, int T_num, int eq_steps, char *filename);
void print_point_2D(double ***arr, int n);

#endif