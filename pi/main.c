#include <stdio.h>
#include <stdbool.h>

#include <CL/opencl.h>
#include <time.h>

#include "ocl_utils.h"
#include "time_utils.h"

double estimate_pi(cl_uint n);

int main(int argc, char *argv[])
{
    if (argc < 1)
    {
        printf("%s <number of runs>\n", argv[0]);
        return -1;
    }
    int n = atoi(argv[1]);
    if (n < 1)
    {
        printf("%d < 1\n", n);
        return -1;
    }

    cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");

    time_measure_start("total");
    double pi = estimate_pi(n);
    time_measure_stop_and_print("total");

    printf("Pi estimation: %lf\n", pi);
}

cl_mem generateRnd(size_t n)
{
    cl_int error;
    cl_float * host = calloc(n, sizeof(cl_float));
    if (host == NULL) abort();
    for (int i = 0; i < n; i++)
    {
        host[i] = ((float)rand()/(float)RAND_MAX);
    }
    cl_mem dev_inpx = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * n, host, &error);
    ocl_err(clFinish(g_command_queue));
    ocl_err(error);
    free(host);
    return dev_inpx;
}

double estimate_pi(const cl_uint n)
{
    const cl_uint group_size = 64;
    const cl_uint groups = n / group_size;

    srand((unsigned int)time(NULL));

    cl_mem dev_inpx = generateRnd(groups * group_size);
    cl_mem dev_inpy = generateRnd(groups * group_size);

    cl_int error;

    // Result buffer on GPU
    cl_mem dev_result = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * groups, NULL, &error);
    ocl_err(error);

    // Create kernel
    cl_kernel kernel = clCreateKernel(g_program, "estimatepi", &error);
    ocl_err(error);

    // Set kernel arguments
    cl_uint arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_uint), &groups));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_uint), &group_size));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_inpx));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_inpy));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_result));

    // Call kernel
    size_t global_work_sizes[] = {groups};
    time_measure_start("computation");
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, 1, NULL, global_work_sizes, NULL, 0, NULL, NULL));
    ocl_err(clFinish(g_command_queue));
    time_measure_stop_and_print("computation");

    // Read result
    time_measure_start("data_transfer");
    cl_uint * outp = calloc(groups, sizeof(cl_uint));
    ocl_err(clEnqueueReadBuffer(g_command_queue, dev_result, CL_TRUE, 0, sizeof(cl_uint) * groups, outp, 0, NULL, NULL));
    time_measure_stop_and_print("data_transfer");

    time_measure_start("count");
    size_t trues = 0;
    size_t _n = 0;
    for (int i = 0; i < groups; i++)
    {
        trues += outp[i];
        _n += group_size;
    }
    time_measure_stop_and_print("count");
    return 4.0 * ((double)trues / (double)_n);
}
