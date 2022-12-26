#include "rand.h"

static void normalize_probs(double *probs, int n)
{
	double s = 0.0;
	for (int i = 0; i < n; i++)
	{
		s += probs[i];
	}
	for (int i = 0; i < n; i++)
	{
		probs[i] /= s;
	}
}

void rand_init()
{
	srand(time(NULL));
}

int rand_int(int min, int max)
{
	return min + (rand() % (max - min + 1));
}

double rand_double(double min, double max)
{
	return min + (max - min) * ((double)rand() / RAND_MAX);
}

int rand_int_of_values(int *numbers, int n, double *probs)
{
	// normalize probs
	double *norm_probs = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
	{
		norm_probs[i] = probs[i];
	}
	normalize_probs(norm_probs, n);

	double rd = rand_double(0.0, 1.0);
	double base = 0.0;
	for (int i = 0; i < n; i++)
	{
		if (rd < (norm_probs[i] + base) && rd > base)
		{
			return numbers[i];
		}
		base += norm_probs[i];
	}
}

double rand_double_of_values(double *numbers, int n, double *probs)
{
	// normalize probs
	double *norm_probs = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
	{
		norm_probs[i] = probs[i];
	}
	normalize_probs(norm_probs, n);

	double rd = rand_double(0.0, 1.0);
	double base = 0.0;
	for (int i = 0; i < n; i++)
	{
		if (rd < (norm_probs[i] + base) && rd > base)
		{
			return numbers[i];
		}
		base += norm_probs[i];
	}
}
