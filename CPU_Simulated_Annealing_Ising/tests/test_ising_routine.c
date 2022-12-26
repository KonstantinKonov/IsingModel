#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../libs/rand.h"
#include "../libs/export.h"
#include "../libs/statistics.h"
#include "../libs/ising.h"

void test_initialize_lattice(int n)
{
	char **lattice;

	printf("------------------\n");
	printf("Start testing initialize_lattice\n");
	printf("Initialize with {-1, 1} with prob_up {0.75, 0.25}\n");
	double probs_0[] = {0.75, 0.25};
	int values_0[] = {-1, 1};
	initialize_lattice(&lattice, n, values_0, probs_0, 2);
	print_lattice(lattice, n);

	printf("Average value %lf, expected value near -0.5\n", calc_mag(lattice, n));

	free(lattice);

	printf("Initialize with {-2, -1, 0, 1, 2} with prob_up {0.2, 0.2, 0.2, 0.2, 0.2}\n");
	double probs_1[] = {0.2, 0.2, 0.2, 0.2, 0.2};
	int values_1[] = {-2, -1, 0, 1, 2};
	initialize_lattice(&lattice, n, values_1, probs_1, 5);
	print_lattice(lattice, n);

	printf("Average value %lf, expected value near 0.0\n", calc_mag(lattice, n));

	free(lattice);

	printf("End testing initialize_lattice\n");
}

void test_make_step(int n)
{
	char **lattice;

	printf("------------------\n");
	printf("Start testing make_step\n");
	printf("Initialize with {-1, 1} with prob_up {0.5, 0.5}\n");
	int values_0[] = {-1, 1};
	double probs_0[] = {0.5, 0.5};
	initialize_lattice(&lattice, n, values_0, probs_0, 2);
	long long energy = calc_energy(lattice, n);
	double mag = calc_mag(lattice, n);
	printf("Lattice before the step:\n");
	print_lattice(lattice, n);
	printf("Energy before the step: %lld\n", energy);
	printf("Magnetization before the step: %lf\n", mag);
	make_step(lattice, n, 1.0, &energy, &mag);
	printf("Lattice after the step:\n");
	print_lattice(lattice, n);
	printf("Energy after the step: %lld calc_energy shows: %lld\n", energy, calc_energy(lattice, n));
	printf("Magnetization after the step: %lf calc_mag shows: %lf\n", mag, calc_mag(lattice, n));
	printf("End testing make_step\n");
}

void test_equilibrate(int n, int n_points)
{
	char **lattice;
	// create structure with size 1 because we don't need equilibration statistics in this case
	// n_points = 1 let write only one (initial) point of equilibration
	struct eq e;
	e.n_points = n_points;

	printf("------------------\n");
	printf("Start testing equilibration\n");
	printf("Initialize with {-1, 1} with prob_up {0.2, 0.8}\n");

	int values_0[] = {-1, 1};
	double probs_0[] = {0.2, 0.8};
	initialize_lattice(&lattice, n, values_0, probs_0, 2);

	long long energy = calc_energy(lattice, n);
	double mag = calc_mag(lattice, n);
	int n_steps = 1000;

	printf("Lattice before the equilibration:\n");
	print_lattice(lattice, n);
	printf("Energy before the equilibration: %lld\n", energy);
	printf("Magnetization before the equilibration: %lf\n", mag);

	equilibrate(lattice, n, 5.0, &energy, &mag, n_steps, &e);

	printf("Lattice after the equilibration (%d steps):\n", n_steps);
	print_lattice(lattice, n);
	printf("Energy after the equilibration: %lld calc_energy shows:%lld\n", energy, calc_energy(lattice, n));
	printf("Magnetization after the equilibration: %lf calc_mag shows:%lf\n", mag, calc_mag(lattice, n));

	printf("Energy array:\n");
	for (int i = 0; i < n_points; i++)
	{
		printf("%lld ", e.en_eq[i]);
	}
	printf("\n");

	printf("Magnetization array:\n");
	for (int i = 0; i < n_points; i++)
	{
		printf("%.2lf ", e.mag_eq[i]);
	}
	printf("\n");

	printf("Ising lattice array:\n");
	for (int i = 0; i < n_points; i++)
	{
		print_lattice(e.lattice[i], n);
		printf("\n");
	}

	printf("--------------------\n");
	printf("Test equilibration at scale\n");
	char **lattice_0;
	int n_0 = 1000;
	int m_0 = 2;
	int values[] = {-1, 1};
	double probs[] = {0.5, 0.5};
	initialize_lattice(&lattice_0, n_0, values_0, probs_0, m_0);
	long long energy_0 = calc_energy(lattice_0, n_0);
	double mag_0 = calc_mag(lattice_0, n_0);
	double T_0 = 3.0;
	int n_steps_0 = 100000000;
	struct eq e_0;
	e_0.n_points = 10;
	printf("Energy before equilibration: %lld\n", energy_0);
	printf("Equilibrate at temperature %lf with %d steps...\n", T_0, n_steps_0);
	equilibrate(lattice_0, n_0, T_0, &energy_0, &mag_0, n_steps_0, &e_0);
	printf("Energy after equilibration: %lld\tcalc_energy shows:%lld\n", energy_0, calc_energy(lattice_0, n_0));
	printf("Magnetization after equilibration: %lf\tcalc_mag shows:%lf\n", mag_0, calc_mag(lattice_0, n_0));

	printf("energy array: ");
	for (int i = 0; i < e_0.n_points; i++)
	{
		printf("%lld ", e_0.en_eq[i]);
	}
	printf("\n");

	printf("magnetization array: ");
	for (int i = 0; i < e_0.n_points; i++)
	{
		printf("%lf ", e_0.mag_eq[i]);
	}
	printf("\n");

	printf("End testing equilibration\n");
}

void test_average()
{
	printf("------------------\n");
	printf("Start testing average\n");
	printf("Initialize with {-1, 1} with prob_up {0.2, 0.8}\n");

	char **lattice_0;
	int n_0 = 1000;
	int values_0[] = {-1, 1};
	double probs_0[] = {0.2, 0.8};
	int m_0 = 2;
	struct eq e_0;
	e_0.n_points = 10;
	struct metrics met;
	long long energy_0 = 0;
	double mag_0 = 0.0;
	double T_0 = 2.27;
	int eq_steps = 100000, avg_steps = 1000;
	initialize_lattice(&lattice_0, n_0, values_0, probs_0, m_0);
	energy_0 = calc_energy(lattice_0, n_0);
	mag_0 = calc_mag(lattice_0, n_0);
	equilibrate(lattice_0, n_0, T_0, &energy_0, &mag_0, eq_steps, &e_0);
	average(lattice_0, n_0, T_0, &energy_0, &mag_0, avg_steps, &met);

	printf("energy=%lld\n", energy_0);
	printf("cv=%lf\n", met.cv);
	printf("End testing average\n");
}

void test_sweep_temp(int n)
{
	printf("------------------\n");
	printf("Start testing sweep_temp\n");
	double *T_arr;
	int T_num = 31;
	int eq_steps = n * n * 100, avg_steps = 1000;
	struct eq e;
	e.n_points = 10;
	struct metrics met;

	linspace(&T_arr, 3.2, 1.5, T_num);

	int values[] = {-1, 1};
	double probs[] = {0.5, 0.5};
	char **lattice;
	long long energy = 0;
	double mag = 0.0;
	initialize_lattice(&lattice, n, values, probs, 2);
	sweep_temp(n, T_arr, T_num, eq_steps, avg_steps, &e, &met);
	printf("End testing sweep_temp\n");
}

void equilibrate_lattice(int n)
{
	long long eq_steps = n * n * 1000;
	struct eq e;
	e.n_points = 100;
	int values[] = {-1, 1};
	double probs[] = {0.5, 0.5};
	long long energy = 0;
	double mag = 0.0;
	char **lattice;
	initialize_lattice(&lattice, n, values, probs, 2);
	energy = calc_energy(lattice, n);
	mag = calc_mag(lattice, n);

	int nn = 21;
	double *T_arr;
	linspace(&T_arr, 3.5, 0.1, nn);

	for (int j = 0; j < nn; j++)
	{
		printf("T=%.2lf\n", T_arr[j]);
		equilibrate(lattice, n, T_arr[j], &energy, &mag, eq_steps, &e);

		for (int i = 0; i < e.n_points; i++)
		{
			char temp[100];
			char filename[100] = "lattice_eq/lattice";
			sprintf(temp, "%d", 100 * j + i);
			strcat(filename, temp);
			export_lattice(e.lattice[i], n, filename);

			// printf("mag: %lf\n", e.mag_eq[i]);
		}
	}
	// export_arr_long_long(e.en_eq, e.n_points, "lattice_eq/en_eq.bin");
	// export_arr_double(e.mag_eq, e.n_points, "lattice_eq/mag_eq.bin");
}

int main(void)
{
	rand_init();
	// test_initialize_lattice(10);
	// test_make_step(5);
	// test_equilibrate(5, 5);
	// test_average();
	test_sweep_temp(64);
	// equilibrate_lattice(500);

	return 0;
}