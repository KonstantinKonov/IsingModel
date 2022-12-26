#include "../libs/statistics.h"
#include "../libs/export_routine.h"
#include "../libs/ising_routine.h"
#include "../libs/rand_routine.h"

void test_linspace()
{
	printf("--------------\n");
	printf("Start testing linspace function\n");

	double *arr_0;
	int n_0 = 11;
	double min_0 = 0.0, max_0 = 10.0;
	printf("Testing linspace from %lf to %lf, num points: %d\n", min_0, max_0, n_0);
	linspace(&arr_0, min_0, max_0, n_0);
	for (int i = 0; i < n_0; i++)
	{
		printf("%.2lf ", arr_0[i]);
	}
	printf("\n");

	double *arr_1;
	int n_1 = 11;
	double min_1 = -10.0, max_1 = 10.0;
	printf("Testing linspace from %lf to %lf, num points: %d\n", min_1, max_1, n_1);
	linspace(&arr_1, min_1, max_1, n_1);
	for (int i = 0; i < n_1; i++)
	{
		printf("%.2lf ", arr_1[i]);
	}
	printf("\n");

	double *arr_2;
	int n_2 = 31;
	double min_2 = 1.5, max_2 = 3.2;
	printf("Testing linspace from %lf to %lf, num points: %d\n", min_2, max_2, n_2);
	linspace(&arr_2, min_2, max_2, n_2);
	for (int i = 0; i < n_2; i++)
	{
		printf("%.2lf ", arr_2[i]);
	}
	printf("\n");

	double *arr_3;
	int n_3 = 31;
	double min_3 = 3.2, max_3 = 1.5;
	printf("Testing linspace from %lf to %lf, num points: %d\n", min_3, max_3, n_3);
	linspace(&arr_3, min_3, max_3, n_3);
	for (int i = 0; i < n_3; i++)
	{
		printf("%.2lf ", arr_3[i]);
	}
	printf("\n");

	printf("End testing linspace function\n");
}

void test_flatten(int n)
{
	printf("--------------\n");
	printf("Start testing flatten function\n");
	char **arr = (char **)malloc(n * sizeof(char *));
	for (int i = 0; i < n; i++)
	{
		arr[i] = (char *)malloc(n * sizeof(char));
	}
	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			arr[x][y] = x + y * n;
		}
	}

	char *res;

	print_lattice(arr, n);

	flatten(arr, n, &res);

	for (int i = 0; i < n * n; i++)
	{
		printf("%d ", res[i]);
	}
	printf("\n");
	printf("End testing flatten function\n");
}

void test_get_neighbors_sum()
{
	printf("--------------\n");
	printf("Start testing get_neighbors_sum function\n");
	char **arr = (char **)malloc(3 * sizeof(char *));
	for (int i = 0; i < 3; i++)
	{
		arr[i] = (char *)malloc(3 * sizeof(char));
	}
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			arr[x][y] = 1;
		}
	}
	printf("Correct answer is 4: %d\n", get_neighbors_sum(arr, 3, 1, 1));
	printf("Correct answer is 4: %d\n", get_neighbors_sum(arr, 3, 0, 0));
	printf("Correct answer is 4: %d\n", get_neighbors_sum(arr, 3, 2, 2));

	arr[2][1] = -1;
	printf("Correct answer is 2: %d\n", get_neighbors_sum(arr, 3, 2, 2));

	printf("End testing get_neighbors_sum function\n");
}

void test_calc_energy()
{
	printf("--------------\n");
	printf("Start testing calc_energy function\n");
	char **arr = (char **)malloc(3 * sizeof(char *));
	for (int i = 0; i < 3; i++)
	{
		arr[i] = (char *)malloc(3 * sizeof(char));
	}
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			arr[x][y] = 1;
		}
	}
	print_lattice(arr, 3);
	printf("Correct answer is -18: %ld\n", calc_energy(arr, 3));
	arr[1][1] = -1;
	print_lattice(arr, 3);
	printf("Correct answer is -10: %ld\n", calc_energy(arr, 3));

	printf("Test scalability\n");
	char **lattice_0;
	int values_0[] = {-1, 1};
	double probs_0[] = {0, 1.0};
	int m_0 = 2;
	int n_0 = 10;
	initialize_lattice(&lattice_0, n_0, values_0, probs_0, m_0);
	printf("Energy should be -1 * 10 * 10/2 = -200\t%ld\n", calc_energy(lattice_0, n_0));

	char **lattice_1;
	int values_1[] = {-1, 1};
	double probs_1[] = {0.2, 0.8};
	int m_1 = 2;
	int n_1 = 3;
	initialize_lattice(&lattice_1, n_1, values_1, probs_1, m_1);
	print_lattice(lattice_1, n_1);
	printf("Energy: %ld\n", calc_energy(lattice_1, n_1));

	printf("End testing calc_energy function\n");
}

void test_calc_mag()
{
	printf("--------------\n");
	printf("Start testing calc_mag function\n");
	char **arr = (char **)malloc(3 * sizeof(char *));
	for (int i = 0; i < 3; i++)
	{
		arr[i] = (char *)malloc(3 * sizeof(char));
	}
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			arr[x][y] = 1;
		}
	}
	print_lattice(arr, 3);
	printf("Correct answer is 9/9 = 1: %lf\n", calc_mag(arr, 3));
	arr[1][1] = -1;
	print_lattice(arr, 3);
	printf("Correct answer is 7/9 = 0.77: %lf\n", calc_mag(arr, 3));
	printf("\n");

	char **lattice_0;
	int n_0 = 1000;
	int m_0 = 2;
	int values_0[] = {-1, 1};
	double probs_0[] = {0.2, 0.8};
	initialize_lattice(&lattice_0, n_0, values_0, probs_0, m_0);
	printf("Magnetization should be close to 0.6\t%lf\n", calc_mag(lattice_0, n_0));
	printf("\n");

	printf("End testing calc_mag function\n");
}

void test_dE()
{
	printf("--------------\n");
	printf("Start testing dE function\n");
	char **arr = (char **)malloc(3 * sizeof(char *));
	for (int i = 0; i < 3; i++)
	{
		arr[i] = (char *)malloc(3 * sizeof(char));
	}
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			arr[x][y] = 1;
		}
	}

	printf("\n");
	print_lattice(arr, 3);
	printf("Energy: %ld\n", calc_energy(arr, 3));
	printf("Changed to\n");
	printf("dE shows: %ld\n", dE(arr, 3, 1, 1));
	arr[1][1] = -arr[1][1];
	print_lattice(arr, 3);
	printf("Energy: %ld\n", calc_energy(arr, 3));
	printf("Changed to\n");
	printf("dE shows: %ld\n", dE(arr, 3, 0, 0));
	arr[0][0] = -arr[0][0];
	print_lattice(arr, 3);
	printf("Energy: %ld\n", calc_energy(arr, 3));
	printf("\n");

	printf("At large scale\n");
	char **lattice_0;
	int n_0 = 1000;
	int m_0 = 2;
	int values_0[] = {-1, 1};
	double probs_0[] = {0.5, 0.5};
	initialize_lattice(&lattice_0, n_0, values_0, probs_0, m_0);
	long start_energy_0 = calc_energy(lattice_0, n_0);
	long dE_acc = 0.0;
	printf("Start calculated energy: %ld\n", start_energy_0);
	printf("Making steps...\n");
	for (int i = 0; i < 1000; i++)
	{
		int x = rand_int(0, n_0 - 1);
		int y = rand_int(0, n_0 - 1);
		dE_acc += dE(lattice_0, n_0, x, y);
		lattice_0[x][y] = -lattice_0[x][y];
	}
	long end_energy_0 = calc_energy(lattice_0, n_0);
	printf("End calculated energy: %ld\n", end_energy_0);
	printf("Delta energy: %ld\n", end_energy_0 - start_energy_0);
	printf("Accumulated dE: %ld\n", dE_acc);

	printf("End testing dE function\n");
}

int main(void)
{
	rand_init();
	// test_linspace();
	// test_flatten(10);
	// test_get_neighbors_sum();
	// test_calc_energy();
	// test_calc_mag();
	// test_dE();

	return 0;
}
