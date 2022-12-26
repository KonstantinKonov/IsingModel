#ifndef __ISING_ROUTINE_H__
#define __ISING_ROUTINE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "export.h"
#include "statistics.h"
#include "rand.h"

/*
structure eq - stores the historical data of energy, magnetization, lattice itself during equilibration
memory for arrays allocated in equilibration function
en_eq - total energy (not divided by amount of spins)
mag_eq - total magnetization (divided by amount of spins)
lattice - ising lattice, pointer to 2D array of char
n_points - amount of data points to store
*/
struct eq
{
	long long *en_eq;
	double *mag_eq;
	char ***lattice;
	int n_points;
};

/*
structure metrics - stores values of specific heat, magnetization, energy, magnetic susceptibility
*/
struct metrics
{
	double cv;
	double mag;
	double energy;
	double susc;
};

/*
initialize lattice
lattice - ising lattice, pointer to 2D array (not allocated)
n - dimension of ising lattice (n x n)
values - 1D array from which values randomly to be taken (e.g. {-1, 1} for simplest case)
probs - 1D array from which probabilities to be taken (accordingly to values)
m - dimension of 'values' and 'probs' arrays
*/
void initialize_lattice(char ***lattice, int n, int *values, double *probs, int m);

/*
make Monte-Carle step
lattice - ising lattice, 2D array
n - dimension of ising lattice (n x n)
T - temperature
energy - a pointer to energy variable
mag - a pointer to magnetization variable
*/
void make_step(char **lattice, int n, double T, long long *energy, double *mag);

/*
equilibration of lattice
lattice - ising lattice, 2D array
n - dimension of ising lattice (n x n)
T - temperature
energy - a pointer to energy variable
mag - a pointer to magnetization variable
n_steps - amount of steps to equilibrate the lattice
e - pointer to structure with historical data
*/
void equilibrate(char **lattice, int n, double T, long long *energy, double *mag, long long n_steps, struct eq *e);

/*
aquisition of metrics parameters
lattice - ising lattice, 2D array
n - dimension of ising lattice (n x n)
T - temperature
energy - a pointer to energy variable
mag - a pointer to magnetization variable
n_steps - amount of steps to equilibrate the lattice
met - pointer to structure with metric data
 */
void average(char **lattice, int n, double T, long long *energy, double *mag, long long n_steps, struct metrics *met);

/*
sweeping temperature - function realizes the outer cycle over the equilibrate and average
n - dimension of ising lattice (n x n)
T_arr - 1D array of temperatures to sweep
T_num - size of T_arr
eq_steps - equilibration steps
avg_steps - averaging steps
e - pointer to structure with historical data
met - pointer to structure with metric data
 */
void sweep_temp(int n, double *T_arr, int T_num, int eq_steps, int avg_steps, struct eq *e, struct metrics *met);

#endif