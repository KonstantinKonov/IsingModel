#include "export.h"

void print_lattice(char **lattice, int n)
{
    for (int y = 0; y < n; y++)
    {
        for (int x = 0; x < n; x++)
        {
            printf("%2d ", lattice[x][y]);
        }
        printf("\n");
    }
}

void export_lattice(char **lattice, int n, char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");
    char *res;
    flatten(lattice, n, &res);
    fwrite(res, n * n * sizeof(char), 1, fp);
    fclose(fp);
}

void export_arr_int(int *arr, int n, char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");
    fwrite(arr, n * sizeof(int), 1, fp);
    fclose(fp);
}

void export_arr_long_long(long long *arr, int n, char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");
    fwrite(arr, n * sizeof(long long), 1, fp);
    fclose(fp);
}

void export_arr_double(double *arr, int n, char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");
    fwrite(arr, n * sizeof(double), 1, fp);
    fclose(fp);
}

void export_arr_double_2D(double **arr, int n, int m, char *filename)
{
    double *res;
    flatten_double(arr, n, m, &res);

    FILE *fp;
    fp = fopen(filename, "w");
    fwrite(res, n * m * sizeof(double), 1, fp);
    fclose(fp);
}