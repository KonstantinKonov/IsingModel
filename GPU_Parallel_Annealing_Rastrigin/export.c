#include "export.h"

void export_double_2D(double **arr, int n, char *filename)
{
	double *res = (double *)malloc(n * n * sizeof(double));

	flatten_double_2D(arr, n, &res);

	FILE *fp = fopen(filename, "w");
	fwrite(res, n * n * sizeof(double), 1, fp);
	fclose(fp);
}

void export_double_1D(double *arr, int n, char *filename)
{
	FILE *fp = fopen(filename, "w");
	fwrite(arr, n * sizeof(double), 1, fp);
	fclose(fp);
}

void export_point_2D(double ***arr, int T_num, int eq_steps, char *filename)
{
	double *res;

	flatten_point_arr_to_double(arr, T_num, eq_steps, &res);

	FILE *fp = fopen(filename, "w");
	fwrite(res, T_num * eq_steps * DIM * sizeof(double), 1, fp);
	fclose(fp);
}

void print_point_2D(double ***arr, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("(");
			for (int k = 0; k < DIM; k++)
			{
				printf("%2.1lf ", arr[i][j][k]);
			}
			printf(")");
		}
		printf("\n");
	}
}