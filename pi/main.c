#include <stdio.h>
#include <stdbool.h>

#include <CL/opencl.h>
#include <time.h>

#include "ocl_utils.h"
#include "time_utils.h"

double estimate_pi(cl_uint n);

int main(int argc, char *argv[])
{
    cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");

    time_measure_start("total");
    double pi = estimate_pi(10000000);
    time_measure_stop_and_print("total");

    printf("Pi estimation: %lf\n", pi);
}

double estimate_pi(cl_uint n)
{
    cl_float * inp = calloc(2 * n, sizeof(cl_float));
    cl_uint * outp = calloc(n, sizeof(cl_uint));

    srand((unsigned int)time(NULL));
    for (int i = 0; i < 2 * n; i++) inp[i] = ((float)rand()/(float)(RAND_MAX));

    cl_int error;

    // Input buffer on GPU
    cl_mem dev_inp = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 2 * n, inp, &error);
    ocl_err(error);
    // Result buffer on GPU
    cl_mem dev_result = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * n, NULL, &error);
    ocl_err(error);

    ocl_err(clFinish(g_command_queue));

    // Create kernel
    cl_kernel kernel = clCreateKernel(g_program, "estimatepi", &error);
    ocl_err(error);

    // Set kernel arguments
    cl_uint arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_uint), &n));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_inp));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_result));

    // Call kernel
    size_t global_work_sizes[] = {n};
    time_measure_start("computation");
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, 1, NULL, global_work_sizes, NULL, 0, NULL, NULL));
    ocl_err(clFinish(g_command_queue));
    time_measure_stop_and_print("computation");

    // Read result
    time_measure_start("data_transfer");
    ocl_err(clEnqueueReadBuffer(g_command_queue, dev_result, CL_TRUE, 0, sizeof(cl_uint) * n, outp, 0, NULL, NULL));
    time_measure_stop_and_print("data_transfer");

    time_measure_start("count");
    size_t trues = 0;
    for (int i = 0; i < n; i++) if (outp[i]) trues++;
    time_measure_stop_and_print("count");
    return 2.0 * (trues / (double)n);
}
