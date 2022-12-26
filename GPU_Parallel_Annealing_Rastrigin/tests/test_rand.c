#include <stdio.h>
#include <stdlib.h>
#include "../rand.h"

void test_rand_int()
{
	printf("---------------------\n");
	printf("Start testing rand_int\n");

	printf("Random numbers from 0 to 1\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%d ", rand_int(0, 1));
	}

	printf("Random numbers from 0 to 10\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%d ", rand_int(0, 10));
	}
	printf("\n");

	printf("Random numbers from -10 to 10\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%d ", rand_int(-10, 10));
	}
	printf("\n");

	printf("End testing rand_int\n");
}

void test_rand_double()
{
	printf("---------------------\n");
	printf("Start testing rand_double\n");

	printf("Random numbers from 0.0 to 1.0\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%lf ", rand_double(0.0, 1.0));
	}
	printf("\n");

	printf("Random numbers from 0.0 to 10.0\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%lf ", rand_double(0.0, 10.0));
	}
	printf("\n");

	printf("Random numbers from -100.0 to 100.0\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%lf ", rand_double(-100.0, 100.0));
	}
	printf("\n");

	printf("End testing rand_double\n");
}

void test_draw_new_x()
{
	printf("---------------------\n");
	printf("Start testing draw_new_x\n");

	printf("Get 10 new x from x =0.0 with step 10.0, border (-20, 20)\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%lf ", draw_new_x(0.0, 10.0, -20.0, 20.0));
	}
	printf("\n");

	printf("Get 10 new x from x =0.0 with step 10.0, border (-5, 5)\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%lf ", draw_new_x(0.0, 10.0, -5.0, 5.0));
	}
	printf("\n");

	printf("End testing draw_new_x\n");
}

int main(void)
{
	rand_init();
	test_rand_int();
	// test_rand_double();
	// test_draw_new_x();
}
