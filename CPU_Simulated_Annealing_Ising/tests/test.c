#include <stdio.h>
#include <stdlib.h>

struct metrics
{
    double *res;
};

void allocate_memory_here(struct metrics *m)
{
    m->res = realloc(m->res, sizeof(double) * 10);
}

int main(void)
{
    struct metrics m;
    for (int i = 0; i < 10; i++)
    {
        allocate_memory_here(&m);
    }
}