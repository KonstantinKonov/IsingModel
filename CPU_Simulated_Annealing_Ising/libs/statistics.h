#ifndef __STATISTICS_H__
#define __STATISTICS_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "statistics.h"
#include "export.h"

/*
linsapce from min to max, number of points n
arr - pointer to 1D array (not allocated)
min - minimum value of linspace
max - maximum value of linspace
n - size of 1D array arr
*/
void linspace(double **arr, double min, double max, int n);

/*
flatten the 2D array to 1D array row by row
lattice - 2D array of ising lattice of type char
n - size of ising lattice (n x n)
res - pointer to 1D array of type char (not allocated)
*/
void flatten(char **lattice, int n, char **res);

/*
flatten the double 2D array to 1D array row by row
arr - 2D array of type double
n - size of arr (n x m)
m - size of arr (n x m)
res - pointer to 1D array of type double (not allocated)
*/
void flatten_double(double **arr, int n, int m, double **res);

/*
get sum of neighbors of ising lattice
lattice - 2D array of ising lattice of type char
n - size of ising lattice (n x n)
x - x coordinate of selected spin
y - y coordinate of selected spin
return - sum of 4 neighbors spin values
*/
int get_neighbors_sum(char **lattice, int n, int x, int y);

/*
calculate energy of ising lattice
lattice - 2D array of ising lattice of type char
n - size of ising lattice (n x n)
return - total energy of ising lattice
*/
long long calc_energy(char **lattice, int n);

/*
calculate magnetization of ising lattice
lattice - 2D array of ising lattice of type char
n - size of ising lattice (n x n)
return - total magnetization of ising lattice
*/
double calc_mag(char **lattice, int n);

/*
calculate delta energy
lattice - 2D array of ising lattice of type char
n - size of ising lattice (n x n)
x - x coordinate of selected spin
y - y coordinate of selected spin
return - delta energy
*/
long dE(char **lattice, int n, int x, int y);

#endif
