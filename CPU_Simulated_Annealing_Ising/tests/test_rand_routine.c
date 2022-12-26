#include <stdio.h>
#include <stdlib.h>
#include "../libs/rand_routine.h"

void test_rand_int(int n)
{
	printf("----------------\n");
	printf("Start testing rand_int function\n");
	printf("Random integers from 0 to 10\n");
	for (int i = 0; i < n; i++)
	{
		printf("%d ", rand_int(0, 10));
	}
	printf("\n");
	printf("----------------\n");
	printf("Random integers from -10 to 10\n");
	for (int i = 0; i < n; i++)
	{
		printf("%d ", rand_int(-10, 10));
	}
	printf("\n");
	printf("----------------\n");
	printf("Random integers from 100 to 200\n");
	for (int i = 0; i < n; i++)
	{
		printf("%d ", rand_int(100, 200));
	}
	printf("\n");
	printf("----------------\n");
	printf("Average of 100 random number from 0 to 100\n");
	int s = 0;
	for (int i = 0; i < n; i++)
	{
		s += rand_int(0, 100);
	}
	printf("%lf\n", (double)s / n);
	printf("End testing rand_int function\n");
}

void test_rand_double(int n)
{
	printf("----------------\n");
	printf("Start testing rand_double function\n");
	printf("Random double from 0 to 10\n");
	for (int i = 0; i < n; i++)
	{
		printf("%lf ", rand_double(0.0, 10.0));
	}
	printf("\n");
	printf("----------------\n");
	printf("Random double from -10 to 10\n");
	for (int i = 0; i < n; i++)
	{
		printf("%lf ", rand_double(-10.0, 10.0));
	}
	printf("\n");
	printf("----------------\n");
	printf("Random double from 100 to 200\n");
	for (int i = 0; i < n; i++)
	{
		printf("%lf ", rand_double(100.0, 200.0));
	}
	printf("\n");
	printf("----------------\n");
	printf("Average of 100 random number from 0 to 100\n");
	int s = 0;
	for (int i = 0; i < n; i++)
	{
		s += rand_double(0, 100);
	}
	printf("%lf\n", (double)s / n);
	printf("End testing rand_double function\n");
}

void test_rand_int_of_values(int n)
{
	printf("----------------\n");
	printf("Start testing rand_int_of_values function\n");
	printf("Get random values from {0, 1} with probabilities {0.5, 0.5}\n");
	int arr_0[] = {0, 1};
	double prob_0[] = {0.5, 0.5};
	for (int i = 0; i < n; i++)
	{
		printf("%d ", rand_int_of_values(arr_0, 2, prob_0));
	}
	printf("Average value of {0, 1} with probabilities {0.5, 0.5}\n");
	int s_0 = 0.0;
	for (int i = 0; i < n; i++)
	{
		s_0 += rand_int_of_values(arr_0, 2, prob_0);
	}
	printf("Average values: %lf\n", (double)s_0 / n);

	printf("Get random values from {-1, 1} with probabilities {0.75, 0.25}\n");
	int arr_1[] = {-1, 1};
	double prob_1[] = {0.75, 0.25};
	for (int i = 0; i < n; i++)
	{
		printf("%d ", rand_int_of_values(arr_1, 2, prob_1));
	}
	printf("Average value of {-1, 1} with probabilities {0.75, 0.25}\n");
	int s_1 = 0.0;
	for (int i = 0; i < n; i++)
	{
		s_1 += rand_int_of_values(arr_1, 2, prob_1);
	}
	printf("Average values: %lf\n", (double)s_1 / n);

	printf("Get random values from {-1, 2, 3} with probabilities {0.2, 0.25, 0.55}\n");
	int arr_2[] = {-1, 2, 3};
	double prob_2[] = {0.2, 0.25, 0.55};
	for (int i = 0; i < n; i++)
	{
		printf("%d ", rand_int_of_values(arr_2, 3, prob_2));
	}
	printf("Average value of {-1, 2, 3} with probabilities {0.2, 0.25, 0.55}\n");
	int s_2 = 0.0;
	for (int i = 0; i < n; i++)
	{
		s_2 += rand_int_of_values(arr_2, 3, prob_2);
	}
	printf("Average values: %lf\n", (double)s_2 / n);
	printf("End testing rand_int_of_values function\n");
}

void test_rand_double_of_values(int n)
{
	printf("----------------\n");
	printf("Start testing rand_double_of_values function\n");
	printf("Get random values from {0, 1} with probabilities {0.5, 0.5}\n");
	double arr_0[] = {0.0, 1.0};
	double prob_0[] = {0.5, 0.5};
	for (int i = 0; i < n; i++)
	{
		printf("%.2lf ", rand_double_of_values(arr_0, 2, prob_0));
	}
	printf("\n");
	printf("Average value of {0, 1} with probabilities {0.5, 0.5}\n");
	double s_0 = 0.0;
	for (int i = 0; i < n; i++)
	{
		s_0 += rand_double_of_values(arr_0, 2, prob_0);
	}
	printf("Average values: %lf\n", s_0 / n);

	printf("Get random values from {-1, 1} with probabilities {0.75, 0.25}\n");
	double arr_1[] = {-1.0, 1.0};
	double prob_1[] = {0.75, 0.25};
	for (int i = 0; i < n; i++)
	{
		printf("%.2lf ", rand_double_of_values(arr_1, 2, prob_1));
	}
	printf("\n");
	printf("Average value of {-1, 1} with probabilities {0.75, 0.25}\n");
	double s_1 = 0.0;
	for (int i = 0; i < n; i++)
	{
		s_1 += rand_double_of_values(arr_1, 2, prob_1);
	}
	printf("Average values: %lf\n", s_1 / n);

	printf("Get random values from {-1, 2, 3} with probabilities {0.2, 0.25, 0.55}\n");
	double arr_2[] = {-1, 2, 3};
	double prob_2[] = {0.2, 0.25, 0.55};
	for (int i = 0; i < n; i++)
	{
		printf("%.2lf ", rand_double_of_values(arr_2, 3, prob_2));
	}
	printf("\n");
	printf("Average value of {-1, 2, 3} with probabilities {0.2, 0.25, 0.55}\n");
	double s_2 = 0.0;
	for (int i = 0; i < n; i++)
	{
		s_2 += rand_double_of_values(arr_2, 3, prob_2);
	}
	printf("Average values: %lf\n", s_2 / n);
	printf("End testing rand_double_of_values function\n");
}

int main(void)
{
	rand_init();

	// test_rand_int(100);
	// test_rand_double(100);
	// test_rand_int_of_values(100);
	test_rand_double_of_values(100);

	return 0;
}
