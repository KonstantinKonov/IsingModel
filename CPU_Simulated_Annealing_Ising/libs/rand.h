#ifndef __RAND_ROUTINE_H__
#define __RAND_ROUTINE_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "rand.h"

/*
seed random numbers with current time value
*/
void rand_init();

/*
get random integer number from min to max, both included
return: picked random integer number
*/
int rand_int(int min, int max);

/*
get random double number from min to max
return: picked random double number
*/
double rand_double(double, double);

/* get integer number from array 'numbers' randomly with probability 'probs'
 * numbers - 1D array
 * n - length of array 'numbers'
 * probs - 1D array
 * return: picked random number
 */
int rand_int_of_values(int *numbers, int n, double *probs);

/* get double number from array 'numbers' randomly with probability 'probs'
 * numbers - 1D array
 * n - length of array 'numbers' and 'probs'
 * probs - 1D array
 * return: picked random number of type double
 */
double rand_double_of_values(double *numbers, int n, double *probs);

#endif
