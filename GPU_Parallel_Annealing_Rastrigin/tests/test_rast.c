#include <stdio.h>
#include <stdlib.h>
#include "../rast.h"
#include "../rand.h"
#include "../export.h"

void test_get_2D_rast()
{
	printf("-------------------\n");
	printf("Start testing get_2D_rast\n");

	double *pt = (double *)malloc(DIM * sizeof(double));
	for (int k = 0; k < DIM; k++)
	{
		pt[k] = 0.0;
	}
	printf("Rast value in (0, 0 ...): %lf\n", get_2D_rast(pt));

	for (int k = 0; k < DIM; k++)
	{
		pt[k] = 1.0;
	}
	printf("Rast value in (1, 1 ...): %lf\n", get_2D_rast(pt));

	for (int k = 0; k < DIM; k++)
	{
		pt[k] = 1.0;
	}
	pt[0] = 2.0;
	printf("Rast value in (2, 1, 1 ...): %lf\n", get_2D_rast(pt));

	printf("End testing get_2D_rast\n");
}

void test_get_rand_point()
{
	printf("-------------------\n");
	printf("Start testing get_rand_point\n");

	double *pt;
	get_rand_point(&pt);
	printf("left and right borders defined in define section in rast.c\n");
	printf("(");
	for (int k = 0; k < DIM; k++)
	{
		printf("%lf ", pt[k]);
	}
	printf(")");
	printf("\n");

	printf("End testing get_rand_point\n");
}

void test_make_step()
{
	printf("-------------------\n");
	printf("Start testing make_step\n");

	double *pt;
	double T = 5.0;
	get_rand_point(&pt);
	printf("Point before step:\n");
	for (int k = 0; k < DIM; k++)
		printf("%lf ", pt[k]);
	printf("\n");

	make_step(pt, T);
	printf("Point after step:\n");
	for (int k = 0; k < DIM; k++)
		printf("%lf ", pt[k]);
	printf("\n");

	for (int i = 0; i < 1000; i++)
	{
		make_step(pt, T);
	}
	printf("Point after n step:\n");
	for (int k = 0; k < DIM; k++)
		printf("%lf ", pt[k]);
	printf("\n");

	printf("Now with decreasing temperature\n");
	while (T > 0.001)
	{
		for (int i = 0; i < 1000; i++)
		{
			make_step(pt, T);
		}
		T *= 0.999;
	}
	printf("Point after decreasing temperature:\n");
	for (int k = 0; k < DIM; k++)
		printf("%lf ", pt[k]);
	printf("\n");

	printf("End testing make_step\n");
}

void test_sweep_temp()
{
	printf("-------------------\n");
	printf("Start testing sweep_temp\n");

	double *T_arr;
	int T_num = 11;
	int eq_steps = 1000;
	linspace(5.0, 0.001, T_num, &T_arr);
	sweep_temp(T_arr, T_num, eq_steps);

	printf("End testing sweep_temp\n");
}

int main(void)
{
	rand_init();
	test_get_2D_rast();
	test_get_rand_point();
	test_make_step();
	test_sweep_temp();

	return 0;
}