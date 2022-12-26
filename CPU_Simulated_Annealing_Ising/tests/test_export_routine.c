#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <string.h>
#include "../libs/export_routine.h"

void test_print_lattice(int n)
{
	char **arr = (char **)malloc(n * sizeof(char *));
	for (int i = 0; i < n; i++)
	{
		arr[i] = (char *)malloc(n * sizeof(char));
	}
	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			arr[x][y] = x + n * y;
		}
	}

	//-----------------------
	printf("---------------------\n");
	printf("Start testing print_lattice\n");

	print_lattice(arr, n);

	printf("End test print_lattice\n");
}

void test_export_lattice(int n, char *filename)
{
	char **arr = (char **)malloc(n * sizeof(char *));
	for (int i = 0; i < n; i++)
	{
		arr[i] = (char *)malloc(n * sizeof(char));
	}
	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			arr[x][y] = x + n * y;
		}
	}

	//-----------------------
	printf("---------------------\n");
	printf("Start testing export_lattice\n");

	export_lattice(arr, n, filename);

	printf("End test export_lattice\n");
}

void test_export_arr_int(int n, char *filename)
{
	int *arr = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++)
	{
		arr[i] = i;
	}

	//-----------------------
	printf("---------------------\n");
	printf("Start testing export_arr_int\n");
	export_arr_int(arr, n, filename);
	printf("End test export_arr_int\n");
}

void test_export_arr_double(int n, char *filename)
{
	double *arr = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
	{
		arr[i] = i / 10.0;
	}

	//-----------------------
	printf("---------------------\n");
	printf("Start testing export_arr_double\n");
	export_arr_double(arr, n, filename);
	printf("End test export_arr_double\n");
}

void test_export_arr_double_2D(int n, int m, char *filename)
{
	double **arr = (double **)malloc(n * sizeof(double *));
	for (int i = 0; i < n; i++)
	{
		arr[i] = (double *)malloc(m * sizeof(double));
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			arr[i][j] = j + i * m;
		}
	}

	printf("---------------------\n");
	printf("Start testing export_arr_double_2D\n");
	export_arr_double_2D(arr, n, m, filename);
	printf("End testing export_arr_double_2D\n");
}

int main(void)
{
	test_print_lattice(10);

	char output_name[PATH_MAX];

	getcwd(output_name, sizeof(output_name));
	strcat(output_name, "/test_outputs/test_export_lattice_0.bin");
	test_export_lattice(100, output_name);

	getcwd(output_name, sizeof(output_name));
	strcat(output_name, "/test_outputs/test_export_arr_int_0.bin");
	test_export_arr_int(100, output_name);

	getcwd(output_name, sizeof(output_name));
	strcat(output_name, "/test_outputs/test_export_arr_double_0.bin");
	test_export_arr_double(100, output_name);

	getcwd(output_name, sizeof(output_name));
	strcat(output_name, "/test_outputs/test_export_arr_double_2D_0.bin");
	test_export_arr_double_2D(10, 5, output_name);

	return 0;
}
