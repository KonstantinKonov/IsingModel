#include "stat.h"

void flatten_double_2D(double **arr, int n, double **res)
{
	*res = (double *)malloc(n * n * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			(*res)[i * n + j] = arr[i][j];
		}
	}
}

void flatten_point_arr_to_double(double ***arr, int n, int m, double **res)
{
	*res = (double *)malloc(n * m * DIM * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			for (int k = 0; k < DIM; k++)
			{
				(*res)[k + j * DIM + i * m * DIM] = arr[i][j][k];
			}
		}
	}
}

void linspace(double min, double max, int n, double **res)
{
	*res = (double *)malloc(n * sizeof(double));
	double step = (max - min) / (n - 1);
	for (int i = 0; i < n; i++)
	{
		(*res)[i] = min + i * step;
	}
}