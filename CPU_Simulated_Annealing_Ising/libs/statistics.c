#include "statistics.h"

void linspace(double **arr, double min, double max, int n)
{
    *arr = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        (*arr)[i] = min + i * (max - min) / (n - 1);
    }
}

void flatten(char **lattice, int n, char **res)
{
    *res = (char *)malloc(n * n * sizeof(char));
    for (int x = 0; x < n; x++)
    {
        for (int y = 0; y < n; y++)
        {
            (*res)[x + n * y] = lattice[x][y];
        }
    }
}

// x and y should be changed vise versa
void flatten_double(double **arr, int n, int m, double **res)
{
    *res = (double *)malloc(n * m * sizeof(double));
    for (int x = 0; x < n; x++)
    {
        for (int y = 0; y < m; y++)
        {
            (*res)[y + x * m] = arr[x][y];
        }
    }
}

int get_neighbors_sum(char **lattice, int n, int x, int y)
{
    int left = x - 1;
    if (left < 0)
        left = n - 1;
    int right = x + 1;
    if (right > n - 1)
        right = 0;
    int top = y - 1;
    if (top < 0)
        top = n - 1;
    int bottom = y + 1;
    if (bottom > n - 1)
        bottom = 0;
    return lattice[left][y] + lattice[right][y] + lattice[x][top] + lattice[x][bottom];
}

long long calc_energy(char **lattice, int n)
{
    long energy = 0;
    for (int x = 0; x < n; x++)
    {
        for (int y = 0; y < n; y++)
        {
            energy += -lattice[x][y] * get_neighbors_sum(lattice, n, x, y);
        }
    }
    return energy / 2;
}

double calc_mag(char **lattice, int n)
{
    int mag = 0;
    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            mag += lattice[x][y];
    return (double)mag / (n * n);
}

long dE(char **lattice, int n, int x, int y)
{
    return 2 * lattice[x][y] * get_neighbors_sum(lattice, n, x, y);
}