#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "time_utils.h"
#include "ocl_utils.h"

const int VECTOR_SIZE = 1024;

cl_mem create_and_init_vector(void)
{
    cl_int error;
    float *host_vec = malloc(sizeof(cl_float3) * VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; ++i)
        host_vec[i] = i;
    cl_mem dev_vec = clCreateBuffer(g_context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float3) * VECTOR_SIZE, host_vec, &error);
    ocl_err(error);
    ocl_err(clFinish(g_command_queue));
    free(host_vec);
    return dev_vec;
}

cl_mem create_result_buffer(void)
{
    cl_int error;
    float *host_vec = malloc(sizeof(cl_float3) * VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; ++i)
        host_vec[i] = 0;

    cl_mem dev_vec = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float3) * VECTOR_SIZE, host_vec, &error);
    ocl_err(error);
    return dev_vec;
}

cl_float *add_numbers(void)
{
    cl_int error;
    // Create device buffers.
    cl_mem dev_vec_a = create_and_init_vector();
    cl_mem dev_vec_b = create_and_init_vector();
    cl_mem dev_result = create_result_buffer();
    cl_float *host_result = malloc(sizeof(cl_float3) * VECTOR_SIZE);

    // Create kernel
    cl_kernel kernel = clCreateKernel(g_program, "add_numbers", &error);
    ocl_err(error);

    // Set kernel arguments
    int arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_vec_a));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_vec_b));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &VECTOR_SIZE));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_result));

    // Call kernel
    size_t global_work_sizes[] = {VECTOR_SIZE};
    time_measure_start("computation");
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, 1, NULL,
                global_work_sizes, NULL, 0, NULL, NULL));
    ocl_err(clFinish(g_command_queue));
    time_measure_stop_and_print("computation");

    // Read result
    time_measure_start("data_transfer");
    ocl_err(clEnqueueReadBuffer(g_command_queue, dev_result, CL_TRUE,
                0, sizeof(cl_float3) * VECTOR_SIZE, host_result, 0, NULL, NULL));
    time_measure_stop_and_print("data_transfer");
    return host_result;
}

void print_result(cl_float *numbers)
{
    for (int i = 0; i < VECTOR_SIZE; ++i)
    {
        printf("%d: %f, ", i, numbers[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");

    time_measure_start("total");
    cl_float *numbers = add_numbers();
    time_measure_stop_and_print("total");
    print_result(numbers);

}
