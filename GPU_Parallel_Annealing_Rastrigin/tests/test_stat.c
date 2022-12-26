#include "../stat.h"
#include "../rand.h"

void test_flatten_double_2D(int n)
{
	double **arr = (double **)malloc(n * sizeof(double *));
	for (int i = 0; i < n; i++)
	{
		arr[i] = (double *)malloc(n * sizeof(double));
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			arr[i][j] = rand_int(0, 10);
		}
	}

	printf("---------------------\n");
	printf("Start testing flatten_double_2D\n");

	printf("Randomly generated 2D array\n");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%.1lf ", arr[i][j]);
		}
		printf("\n");
	}

	printf("Flattened array\n");
	double *flatten = (double *)malloc(n * n * sizeof(double));
	flatten_double_2D(arr, n, &flatten);

	for (int i = 0; i < n * n; i++)
	{
		printf("%.1lf ", flatten[i]);
	}
	printf("\n");
	printf("End testing flatten_double_2D\n");
}

void test_linspace()
{
	printf("---------------------\n");
	printf("Start testing linspace\n");

	printf("Generating linspace from 0 to 1 with 10 steps\n");
	double *res_0 = (double *)malloc(10 * sizeof(double));
	linspace(0.0, 1.0, 10, &res_0);
	for (int i = 0; i < 10; i++)
	{
		printf("%.2lf ", res_0[i]);
	}
	printf("\n");

	printf("Generating linspace from 0 to 1 with 11 steps\n");
	double *res_1 = (double *)malloc(10 * sizeof(double));
	linspace(0.0, 1.0, 11, &res_1);
	for (int i = 0; i < 11; i++)
	{
		printf("%.2lf ", res_1[i]);
	}
	printf("\n");

	printf("Generating linspace from -1 to 1 with 11 steps\n");
	double *res_2 = (double *)malloc(10 * sizeof(double));
	linspace(-1.0, 1.0, 11, &res_2);
	for (int i = 0; i < 11; i++)
	{
		printf("%.1lf ", res_2[i]);
	}
	printf("\n");

	printf("Generating linspace from 10 to -10 with 51 steps\n");
	double *res_3 = (double *)malloc(10 * sizeof(double));
	linspace(10.0, -10.0, 51, &res_3);
	for (int i = 0; i < 51; i++)
	{
		printf("%.1lf ", res_3[i]);
	}
	printf("\n");
	printf("End testing linspace\n");
}

void test_flatten_point_arr_to_double(int n, int m)
{
	printf("---------------------\n");
	printf("Start testing flatten_point_arr\n");

	printf("Generating 2D array of points\n");
	// point arr
	double ***arr = (double ***)malloc(n * sizeof(double **));
	for (int i = 0; i < n; i++)
	{
		arr[i] = (double **)malloc(m * sizeof(double *));
		for (int j = 0; j < m; j++)
		{
			arr[i][j] = (double *)malloc(DIM * sizeof(double));
			for (int k = 0; k < DIM; k++)
			{
				arr[i][j][k] = rand_int(0, 10);
			}
		}
	}

	printf("Printing 2D array of points\n");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			printf("(");
			for (int k = 0; k < DIM; k++)
			{
				printf("%2.0lf ", arr[i][j][k]);
			}
			printf(")");
		}
		printf("\n");
	}

	printf("Flattening the array\n");
	// point res
	double *res;
	flatten_point_arr_to_double(arr, n, m, &res);
	for (int i = 0; i < n * m; i++)
	{
		printf("(");
		for (int i = 0; i < DIM; i++)
		{
			printf("%2.0lf ", res[i]);
		}
		printf(")");
	}
	printf("\n");

	printf("End testing flatten_point_arr\n");
}

int main(void)
{
	rand_init();
	test_flatten_double_2D(4);
	test_linspace();
	test_flatten_point_arr_to_double(3, 4);

	return 0;
}