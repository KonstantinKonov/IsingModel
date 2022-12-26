#include "rast.h"

__host__ void linspace(double min, double max, int n, double **res)
{
	*res = (double *)malloc(n * sizeof(double));
	double step = (max - min) / (n - 1);
	for (int i = 0; i < n; i++)
	{
		(*res)[i] = min + i * step;
	}
}

__device__ double get_nD_rast(double *pt)
{
	double rast = 10.0 * DIM;
	for (int k = 0; k < DIM; k++)
	{
		rast += pt[k] * pt[k] - 10.0 * cos(2 * M_PI * pt[k]);
	}
	return rast;
}

__device__ double draw_new_x(double x, curandState *crs)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	double step = 2 * curand_uniform(crs + idx) * STEP_SIZE - STEP_SIZE;
	double new_x = x + step;
	if (new_x > BORDER_RIGHT)
	{
		new_x = BORDER_LEFT + (new_x - BORDER_RIGHT);
	}
	if (new_x < BORDER_LEFT)
	{
		new_x = BORDER_RIGHT - (BORDER_LEFT - new_x);
	}

	return new_x;
}

__device__ void get_rand_point(double **pt, curandState *crs)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	*pt = (double *)malloc(DIM * sizeof(double));

	for (int k = 0; k < DIM; k++)
	{
		(*pt)[k] = BORDER_LEFT + curand_uniform(crs + idx) * (BORDER_RIGHT - BORDER_LEFT);
	}
}

__device__ void make_step(double *curr_point, double T, curandState *crs)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int choise = -1 + ceil(curand_uniform(crs + idx) * DIM);
	double prev = curr_point[choise];
	double cand = draw_new_x(prev, crs);
	double dE = -get_nD_rast(curr_point);
	curr_point[choise] = cand;
	dE += get_nD_rast(curr_point);
	if ((dE > 0) && (exp(-dE / T) < curand_uniform(crs + idx)))
	{
		curr_point[choise] = prev;
	}
}

__global__ void init_curand(curandState *crs, long *d_seed)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(*d_seed, idx, 0, &crs[idx]);
}

__global__ void sweep_temp(double *T, curandState *crs, double *points, double *energies, double *stat_sum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// get_rand_point(&curr_point, crs);

	for (int j = 0; j < EQ_STEPS_CHUNK; j++)
	{
		make_step(points + idx * DIM, *T, crs);
	}
	energies[idx] = get_nD_rast(points + idx * DIM);
	energies[idx] = exp(-energies[idx] / *T);
	__syncthreads();

	// block reduce
	__shared__ double cache[BLOCK_SIZE];
	double block_energies[GRID_SIZE];
	int cacheIndex = threadIdx.x;
	cache[cacheIndex] = energies[idx];
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	double stat_sum_local = 0.0;
	if (cacheIndex == 0)
	{
		block_energies[blockIdx.x] = cache[0];
		__syncthreads();
		for (int i = 0; i < GRID_SIZE; i++)
		{
			stat_sum_local += block_energies[i];
		}
	}

	// exchange
	energies[idx] /= stat_sum_local;
	// reduce min
	double min_val = energies[0];
	int min_val_idx = 0;
	if (idx == 0)
	{
		for (int i = 1; i < REPLICS; i++)
		{
			if (energies[i] < min_val)
			{
				min_val = energies[i];
				min_val_idx = i;
			}
		}
	}
	// exchange
	for (int k = 0; k < DIM; k++)
		points[idx * DIM + k] = points[min_val_idx * DIM + k];
}

int main(void)
{
	curandState *d_crs;
	long seed = time(NULL);
	long *d_seed;

	HANDLE_ERROR(cudaMalloc(&d_crs, GRID_SIZE * BLOCK_SIZE * sizeof(curandState)));
	HANDLE_ERROR(cudaMalloc(&d_seed, sizeof(long)));
	HANDLE_ERROR(cudaMemcpy(d_seed, &seed, sizeof(long), cudaMemcpyHostToDevice));

	init_curand<<<GRID_SIZE, BLOCK_SIZE>>>(d_crs, d_seed);
	cudaDeviceSynchronize();

	HANDLE_ERROR(cudaMemcpy(&seed, d_seed, sizeof(long), cudaMemcpyDeviceToHost));

	double *T_arr;
	linspace(T_START, T_END, T_NUM, &T_arr);

	double *d_T;
	HANDLE_ERROR(cudaMalloc(&d_T, sizeof(double)));

	cudaError_t err;
	srand(time(NULL));
	double *points = (double *)malloc(REPLICS * DIM * sizeof(double));
	for (int i = 0; i < REPLICS * DIM; i++)
	{
		points[i] = BORDER_LEFT + (BORDER_RIGHT - BORDER_LEFT) * ((double)rand() / RAND_MAX);
	}

	printf("\n");
	double *d_points;
	HANDLE_ERROR(cudaMalloc(&d_points, REPLICS * DIM * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(d_points, points, REPLICS * DIM * sizeof(double), cudaMemcpyHostToDevice));

	double *energies = (double *)malloc(REPLICS * sizeof(double));
	double *block_energies = (double *)malloc(GRID_SIZE * sizeof(double));
	double *d_energies, *d_stat_sum;

	HANDLE_ERROR(cudaMalloc(&d_energies, REPLICS * sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&d_stat_sum, sizeof(double)));

	for (int i = 0; i < T_NUM; i++)
	{
		printf("Temp: %lf\n", T_arr[i]);
		HANDLE_ERROR(cudaMemcpy(d_T, &(T_arr[i]), sizeof(double), cudaMemcpyHostToDevice));
		for (int j = 0; j < EQ_CHUNK_NUM; j++)
		{
			sweep_temp<<<GRID_SIZE, BLOCK_SIZE>>>(d_T, d_crs, d_points, d_energies, d_stat_sum);
			cudaDeviceSynchronize();
		}
	}
	err = cudaGetLastError();
	printf("sweep_temp: %s\n", cudaGetErrorString(err));

	HANDLE_ERROR(cudaMemcpy(points, d_points, REPLICS * DIM * sizeof(double), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(energies, d_energies, REPLICS * sizeof(double), cudaMemcpyDeviceToHost));

	for (int thread = 0; thread < 3; thread++)
	{
		for (int i = thread * DIM; i < thread * DIM + DIM; i++)
		{
			printf("%lf ", points[i]);
		}
		printf("\n");
	}

	cudaFree(d_points);
	cudaFree(d_crs);
	cudaFree(d_T);
	cudaFree(d_energies);
}