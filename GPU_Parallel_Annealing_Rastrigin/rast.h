#ifndef __RAST_H__
#define __RAST_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "handle_error.h"

#define DIM 5000
#define BORDER_LEFT (-3000.0 * M_PI)
#define BORDER_RIGHT (3000.0 * M_PI)
#define STEP_SIZE 10
#define T_START 5.0
#define T_END 0.001
#define T_NUM 11
#define EQ_STEPS_CHUNK 1000
#define EQ_CHUNK_NUM 10

#define BLOCK_SIZE 192
#define GRID_SIZE 1
#define REPLICS (GRID_SIZE * BLOCK_SIZE)

__device__ double get_nD_rast(double *pt);
__device__ double draw_new_x(double x, curandState *crs);
__device__ void get_rand_point(double **pt, curandState *crs);
__device__ void make_step(double *curr_point, double T, curandState *crs);
__global__ void init_curand(curandState *crs, long *d_seed);
__global__ void sweep_temp(double *T, curandState *crs, double *points, double *energies, double *block_energies);

#endif