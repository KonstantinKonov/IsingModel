#include "rand.h"

void rand_init()
{
	srand(time(NULL));
}

int rand_int(int min, int max)
{
	return min + rand() % (max - min + 1);
}

double rand_double(double min, double max)
{
	return min + ((double)rand() / RAND_MAX) * (max - min);
}

double draw_new_x(double x, double step_size, double border_low, double border_high)
{
	double step = rand_double(-step_size, step_size);
	double new_x = x + step;
	if (new_x > border_high)
	{
		new_x = new_x - border_high;
	}
	if (new_x < border_low)
	{
		new_x = border_high - (border_low - new_x);
	}

	return new_x;
}