#include "ising.h"

void initialize_lattice(char ***lattice, int n, int *values, double *probs, int m)
{
    // allocate memory for ising lattice
    (*lattice) = (char **)malloc(n * sizeof(char *));
    for (int i = 0; i < n; i++)
    {
        (*lattice)[i] = (char *)malloc(n * sizeof(char));
    }

    // populate ising lattice with random values with probabilities probs
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*lattice)[i][j] = rand_int_of_values(values, m, probs);
        }
    }
}

void make_step(char **lattice, int n, double T, long long *energy, double *mag)
{
    // choose target spin randomly
    int x = rand_int(0, n - 1);
    int y = rand_int(0, n - 1);
    // calculate delta energy
    int deltaE = dE(lattice, n, x, y);
    // Metropolis criterion
    if (deltaE <= 0 || exp(-(double)deltaE / T) > rand_double(0.0, 1.0))
    {
        lattice[x][y] *= -1;
        *energy += deltaE;
        *mag += (2 * lattice[x][y]) / (double)(n * n);
    }
}

void equilibrate(char **lattice, int n, double T, long long *energy, double *mag, long long n_steps, struct eq *e)
{
    // allocate memory for historical data
    e->en_eq = (long long *)malloc(e->n_points * sizeof(long long));
    e->mag_eq = (double *)malloc(e->n_points * sizeof(double));
    e->lattice = (char ***)malloc(e->n_points * sizeof(char **));

    int point_cntr = 0;

    for (long long i = 0; i < n_steps; i++)
    {
        // write historical data
        if (i % (n_steps / e->n_points) == 0)
        {
            e->en_eq[point_cntr] = *energy;
            e->mag_eq[point_cntr] = calc_mag(lattice, n);
            e->lattice[point_cntr] = (char **)malloc(n * sizeof(char *));
            for (int i = 0; i < n; i++)
            {
                e->lattice[point_cntr][i] = (char *)malloc(n * sizeof(char));
            }
            for (int x = 0; x < n; x++)
            {
                for (int y = 0; y < n; y++)
                {
                    e->lattice[point_cntr][x][y] = lattice[x][y];
                }
            }

            point_cntr++;
        }

        // make MCMC step
        make_step(lattice, n, T, energy, mag);
    }
}

void average(char **lattice, int n, double T, long long *energy, double *mag, long long n_steps, struct metrics *met)
{
    double en = (double)(*energy);
    double en2 = (double)(*energy) * (double)(*energy);
    double m1 = abs(*mag);
    double m2 = (*mag) * (*mag);

    for (long long i = 0; i < n_steps; i++)
    {
        make_step(lattice, n, T, energy, mag);
        en += (double)(*energy);

        en2 += (double)(*energy) * (double)(*energy);
        m1 += abs(*mag);
        m2 += (*mag) * (*mag);
    }

    en /= (n_steps + 1);
    en2 /= (n_steps + 1);
    m1 /= (n_steps + 1);
    m2 /= (n_steps + 1);

    met->energy = en / (double)(n * n);
    met->mag = *mag;
    met->susc = (m2 - m1 * m1) / (n * n * T);
    met->cv = (en2 - en * en) / (n * n * T * T);
}

void sweep_temp(int n, double *T_arr, int T_num, int eq_steps, int avg_steps, struct eq *e, struct metrics *met)
{
    // hardcoded it for now
    double probs[] = {0.1, 0.9};
    int values[] = {-1, 1};
    int m = 2;

    double *cv_arr = (double *)malloc(T_num * sizeof(double));
    double *en_arr = (double *)malloc(T_num * sizeof(double));

    double *mag_arr = (double *)malloc(T_num * sizeof(double));
    double *susc_arr = (double *)malloc(T_num * sizeof(double));

    char **lattice;
    initialize_lattice(&lattice, n, values, probs, m);
    long long energy = calc_energy(lattice, n);
    double mag = calc_mag(lattice, n);
    equilibrate(lattice, n, T_arr[0], &energy, &mag, n * n * 10000, e);

    for (int i = 0; i < T_num; i++)
    {
        printf("Temperature: %lf\n", T_arr[i]);
        equilibrate(lattice, n, T_arr[i], &energy, &mag, eq_steps, e);
        average(lattice, n, T_arr[i], &energy, &mag, avg_steps, met);
        cv_arr[i] = met->cv;
        en_arr[i] = met->energy;

        mag_arr[i] = met->mag;
        susc_arr[i] = met->susc;
    }
    /*
    export_arr_double(T_arr, T_num, "output_bins/T.bin");
    export_arr_double(cv_arr, T_num, "output_bins/cv.bin");
    export_arr_double(en_arr, T_num, "output_bins/en.bin");

    export_arr_double(mag_arr, T_num, "output_bins/mag.bin");
    export_arr_double(susc_arr, T_num, "output_bins/susc.bin");
    */
}