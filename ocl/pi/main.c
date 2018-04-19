#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "time_utils.h"


void usage(const char *prog_name)
{
    printf("Usage:\n\t");
    printf("%s <num_samples>\n", prog_name);
}

cl_float2 *create_vectors(int num_vectors)
{
    cl_float2 * vectors = (cl_float2 *)malloc(sizeof(cl_float2) * num_vectors);

    for (int i = 0; i < num_vectors; ++i)
    {
        vectors[i].s[0] = (float)rand() / (float)RAND_MAX;
        vectors[i].s[1] = (float)rand() / (float)RAND_MAX;
    }
    return vectors;
}

cl_float2 create_vector(void)
{
    cl_float2 vector;
    vector.s[0] = (float)rand() / (float)RAND_MAX;
    vector.s[1] = (float)rand() / (float)RAND_MAX;

    return vector;
}

bool is_in_unit_circle(cl_float2 vector)
{
    float size_squared = vector.s[0] * vector.s[0] + vector.s[1] * vector.s[1];
    return size_squared < 1.f;
}

float calc_pi(cl_float2 *vectors, int num_samples)
{
    int num_in_unit_circle = 0;
    for (int i = 0; i < num_samples; ++i)
    {
        if (is_in_unit_circle(create_vector()))
            num_in_unit_circle++;
    }
    return ((float)num_in_unit_circle / (float)num_samples) * 4.f;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        usage(argv[0]);
        exit(-1);
    }
    int num_samples = atoi(argv[1]);
    if (num_samples <= 0)
    {
        usage(argv[0]);
        exit(-1);
    }
    srand(time(NULL));

    time_measure_start("total");
    time_measure_start("create vectors");
    cl_float2 * vectors = create_vectors(num_samples);
    time_measure_stop_and_print("create vectors");
    time_measure_start("calculation");
    float pi = calc_pi(NULL, num_samples);
    time_measure_stop_and_print("calculation");
    free(vectors);
    time_measure_stop_and_print("total");

    printf("Calculated PI using %d samples, result is %f\n", num_samples, pi);
}
