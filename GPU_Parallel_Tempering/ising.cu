#include "ising.h"

__host__ void linspace(double min, double max, int n, double **res)
{
	*res = (double *)malloc(n * sizeof(double));
	double step;
	if (n > 1)
		step = (max - min) / (n - 1);
	else
		step = 0;
	for (int i = 0; i < n; i++)
	{
		(*res)[i] = min + i * step;
	}
}

__global__ void init_curand(curandState *crs, long *d_seed)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(*d_seed, idx, 0, &crs[idx]);
}

__device__ void print_lattice(char *lattice, int thread_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx == thread_num)
	{
		for (int i = thread_num * N * N; i < (thread_num + 1) * N * N; i++)
		{
			printf("%2d ", lattice[i]);
			if ((i + 1) % N == 0)
				printf("\n");
		}
		printf("\n");
	}
}

__device__ int get_neighbors_sum(char *lattice, int x, int y)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int left_x = x - 1;
	int left_y = y;
	if (left_x < 0)
		left_x = N - 1;

	int right_x = x + 1;
	int right_y = y;
	if (right_x > N - 1)
		right_x = 0;

	int top_x = x;
	int top_y = y - 1;
	if (top_y < 0)
		top_y = N - 1;

	int bottom_x = x;
	int bottom_y = y + 1;
	if (bottom_y > N - 1)
		bottom_y = 0;

	return lattice[idx * N * N + left_x + left_y * N] + lattice[idx * N * N + right_x + right_y * N] + lattice[idx * N * N + top_x + top_y * N] + lattice[idx * N * N + bottom_x + bottom_y * N];
}

__device__ long calc_energy(char *lattice)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	long E = 0;
	for (int i = 0; i < N * N; i++)
	{
		E += -lattice[idx * N * N + i] * get_neighbors_sum(lattice, i % N, i / N);
	}
	return E;
}

__global__ void init_lattice(char *lattice, curandState *crs, long *energy)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < N * N; i++)
	{
		lattice[idx * N * N + i] = round(curand_uniform(crs + idx)) * 2 - 1;
	}
	__syncthreads();
	energy[idx] = calc_energy(lattice);

	// print_lattice(lattice, 0);
	// if (idx == 0)
	//{
	// for (int i = 0; i < REPLICS; i++)
	//{
	// for (int j = 0; j < N * N; j++)
	//{
	// printf("%2d ", lattice[i * N * N + j]);
	// if ((j + 1) % N == 0)
	//{
	// printf("\n");
	//}
	//}
	// printf("\n");
	// printf("energy: %ld\n", energy[idx]);
	//}
	//}
}

__global__ void equilibrate(char *lattice, double *T_arr, curandState *crs, long *energy)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	long shift = idx * N * N;
	for (int i = 0; i < EQ_CHUNK_SIZE; i++)
	{
		// choose random point
		int point_index = ceil(N * N * curand_uniform(crs + idx)) - 1;
		// calculate delta energy
		long dE = 2 * lattice[shift + point_index] * get_neighbors_sum(lattice, point_index % N, point_index / N);
		// printf("thread: %d\tpoint index: %d\tdE: %ld\ttemp: %lf\n", idx, point_index, dE, T_arr[idx]);
		//  flip spin or not flip spin
		if ((dE <= 0) || (exp(-dE / T_arr[idx]) > curand_uniform(crs + idx)))
		{
			lattice[shift + point_index] *= -1;
			energy[idx] += dE;
		}
	}
}

__device__ void switch_two_replicas(char *lattice, int n1, int n2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx == 0)
	{
		char temp[N * N];
		for (int i = 0; i < N * N; i++)
		{
			temp[i] = lattice[n1 * N * N + i];
		}
		for (int i = 0; i < N * N; i++)
		{
			lattice[n1 * N * N + i] = lattice[n2 * N * N + i];
		}
		for (int i = 0; i < N * N; i++)
		{
			lattice[n2 * N * N + i] = temp[i];
		}
	}
}

__global__ void exchange(char *lattice, double *T_arr, curandState *crs, long *energy)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx == 0)
	{
		int n1, n2;
		for (int i = 0; i < REPLICS; i++)
		{
			n1 = REPLICS * curand_uniform(crs + idx);
			n2 = REPLICS * curand_uniform(crs + idx);
			if (exp((1 / T_arr[n1] - 1 / T_arr[n2]) * (energy[n1] - energy[n2])) > curand_uniform(crs + idx))
			{
				switch_two_replicas(lattice, n1, n2);
			}
		}
	}
}

__host__ void save_temp(double *T_arr, const char *filename)
{
	FILE *fp = fopen(filename, "w");
	fwrite(T_arr, REPLICS * sizeof(double), 1, fp);
	fclose(fp);
}

__host__ void save_energy(long *energy, const char *filename)
{
	FILE *fp = fopen(filename, "w");
	fwrite(energy, REPLICS * sizeof(long), 1, fp);
	fclose(fp);
}

__host__ void save_lattices(char *lattice, const char *filename)
{
	FILE *fp = fopen(filename, "w");
	fwrite(lattice, REPLICS * N * N * sizeof(char), 1, fp);
	fclose(fp);
}

int main(void)
{
	// initialize curand
	long seed = time(NULL);
	long *d_seed;
	curandState *d_crs;
	cudaError_t err;

	HANDLE_ERROR(cudaMalloc(&d_crs, REPLICS * sizeof(curandState)));
	HANDLE_ERROR(cudaMalloc(&d_seed, sizeof(long)));
	HANDLE_ERROR(cudaMemcpy(d_seed, &seed, sizeof(long), cudaMemcpyHostToDevice));

	init_curand<<<GRID_SIZE, BLOCK_SIZE>>>(d_crs, d_seed);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	printf("init_curand: %s\n", cudaGetErrorString(err));

	// initialize lattice with calculation of energy
	char *d_lattice;
	long *d_energy;
	HANDLE_ERROR(cudaMalloc(&d_lattice, REPLICS * N * N * sizeof(char)));
	HANDLE_ERROR(cudaMalloc(&d_energy, REPLICS * sizeof(double)));

	init_lattice<<<GRID_SIZE, BLOCK_SIZE>>>(d_lattice, d_crs, d_energy);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	printf("init_lattice: %s\n", cudaGetErrorString(err));

	// temperature array
	double *T_arr;
	double *d_T_arr;
	linspace(T_HIGH, T_LOW, REPLICS, &T_arr);
	HANDLE_ERROR(cudaMalloc(&d_T_arr, REPLICS * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(d_T_arr, T_arr, REPLICS * sizeof(double), cudaMemcpyHostToDevice));

	// make steps
	for (int i = 0; i < EQ_CHUNK_NUM; i++)
	{
		printf("EQ chunk #: %d\n", i);
		equilibrate<<<GRID_SIZE, BLOCK_SIZE>>>(d_lattice, d_T_arr, d_crs, d_energy);
		err = cudaGetLastError();
		printf("equilibrate: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		exchange<<<GRID_SIZE, BLOCK_SIZE>>>(d_lattice, d_T_arr, d_crs, d_energy);
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		printf("exchange: %s\n", cudaGetErrorString(err));
	}

	char *lattice = (char *)malloc(REPLICS * N * N * sizeof(char));
	long *energy = (long *)malloc(REPLICS * sizeof(long));
	HANDLE_ERROR(cudaMemcpy(lattice, d_lattice, REPLICS * N * N * sizeof(char), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(energy, d_energy, REPLICS * sizeof(long), cudaMemcpyDeviceToHost));
	save_temp(T_arr, "temp.bin");
	save_lattices(lattice, "lattice.bin");
	save_energy(energy, "energy.bin");

	return 0;
}