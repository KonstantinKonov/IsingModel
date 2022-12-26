#ifndef __ISING_H__
#define __ISING_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "handle_error.h"

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "handle_error.h"

#define N 128

#define T_HIGH 5.0
#define T_LOW 0.1

#define BLOCK_SIZE 16
#define GRID_SIZE 1
#define REPLICS (GRID_SIZE * BLOCK_SIZE)

#define EQ_CHUNK_SIZE (int)1E5
#define EQ_CHUNK_NUM (int)1E3

#endif