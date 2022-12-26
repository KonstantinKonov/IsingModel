#ifndef __EXPORT_ROUTINE_H__
#define __EXPORT_ROUTINE_H__

#include <stdio.h>
#include <stdlib.h>

#include "rand.h"
#include "statistics.h"

/*
print ising lattice to stdout
lattice - ising lattice, 2D array of char
n - size of ising lattice (n x n)
*/
void print_lattice(char **lattice, int n);

/*
export ising lattice to binary file
lattice - ising lattice, 2D array of char
n - size of ising lattice (n x n)
filename - name of binary file
*/
void export_lattice(char **lattice, int n, char *filename);

/*
export 1D array of type int to file
arr - 1D array of int
n - size of 1D array
filename - name of binary file
*/
void export_arr_int(int *arr, int n, char *filename);

/*
export 1D array of type double to file
arr - 1D array of double
n - size of 1D array
filename - name of binary file
*/
void export_arr_double(double *arr, int n, char *filename);

/*
export 1D array of type long long to file
arr - 1D array of long long
n - size of 1D array
filename - name of binary file
*/
void export_arr_long_long(long long *arr, int n, char *filename);

/*
export 2D array of type double to file
arr - 2D array of double
n, m - sizes of 2D array
filename - name of binary file
*/
void export_arr_double_2D(double **arr, int n, int m, char *filename);

#endif