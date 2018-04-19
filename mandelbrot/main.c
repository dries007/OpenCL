#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/opencl.h>

#include "ocl_utils.h"
#include "time_utils.h"
#include "image_helper.h"

const int WIDTH = 640 * 10;
const int HEIGHT = 480 * 10;
#define VECTOR_SIZE (WIDTH * HEIGHT)

void mandelbrot(cl_float * output)
{
    cl_int error;

    // Result buffer on GPU
    cl_mem dev_result = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * VECTOR_SIZE, NULL, &error);
    ocl_err(error);

    // Create kernel
    cl_kernel kernel = clCreateKernel(g_program, "mandelbrot", &error);
    ocl_err(error);

    // Set kernel arguments
    cl_uint arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_uint), &WIDTH));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_uint), &HEIGHT));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_result));

    // Call kernel
    size_t global_work_sizes[] = {WIDTH, HEIGHT};
    time_measure_start("computation");
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, 2, NULL, global_work_sizes, NULL, 0, NULL, NULL));
    ocl_err(clFinish(g_command_queue));
    time_measure_stop_and_print("computation");

    // Read result
    time_measure_start("data_transfer");
    ocl_err(clEnqueueReadBuffer(g_command_queue, dev_result, CL_TRUE, 0, sizeof(cl_float) * VECTOR_SIZE, output, 0, NULL, NULL));
    time_measure_stop_and_print("data_transfer");
}

void setRGB(unsigned char *ptr, float val)
{
    ptr[0] = (unsigned char)val;
    ptr[1] = (unsigned char)val;
    ptr[2] = (unsigned char)val;
}

int main(int argc, char *argv[])
{
    cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");

    float * output = calloc(VECTOR_SIZE, sizeof(cl_float));

    time_measure_start("total");
    mandelbrot(output);
    time_measure_stop_and_print("total");

    writeImage("mandelbrot.png", WIDTH, HEIGHT, output, "Mandelbrot by GPU");
}
