#include <stdio.h>
#include <stdlib.h>
#include "libs/export.h"
#include "libs/rand.h"
#include "libs/statistics.h"
#include "libs/ising.h"

#define N 64
#define T_MAX 5.0
#define T_MIN 0.5
#define T_NUM 31

int main(void)
{
	rand_init();

	double *T_arr;
	int T_num = 31;
	int eq_steps = N * N * 100, avg_steps = 1000;
	struct eq e;
	e.n_points = 10;
	struct metrics met;

	linspace(&T_arr, 3.2, 1.5, T_num);

	int values[] = {-1, 1};
	double probs[] = {0.5, 0.5};
	char **lattice;
	long long energy = 0;
	double mag = 0.0;
	initialize_lattice(&lattice, N, values, probs, 2);
	sweep_temp(N, T_arr, T_num, eq_steps, avg_steps, &e, &met);
}