//
//  main.c
//  Dotproduct
//
//  Created by Joost Vennekens on 11/02/15.
//  Copyright (c) 2015 Joost Vennekens. All rights reserved.
//

#include <stdio.h>
#include <CL/opencl.h>
#include <time.h>
#include <math.h>
#include <string.h>

char* get_line(FILE* f);

const char* sourceFile = "./kernel.cl";

cl_context context;
cl_command_queue queue;
cl_program program;
size_t max_local;

void err_check(int error, char* loc) {
    if (error) {
        printf("ERROR: %d when %s\n", error, loc);
        exit(300);
    }
}

cl_kernel OCLPrepKernelMult(int* vec1, int* vec2, int n, cl_mem* result) {
    cl_int error;
    cl_kernel kernel = clCreateKernel(program, "multiply2", &error );
    err_check(error, "creating kernel");
    
    cl_mem ovec1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * n, vec1, &error);
    err_check(error, "creating buffer for first vector");
    clSetKernelArg(kernel, 0, sizeof(ovec1), &ovec1);
    
    cl_mem ovec2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * n, vec2, &error);
    err_check(error, "creating buffer for second vector");
    clSetKernelArg(kernel, 1, sizeof(ovec2), &ovec2);
    
    *result = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(cl_int) * n, NULL, &error);
    err_check(error, "creating buffer for result");
    clSetKernelArg(kernel, 2, sizeof(*result), result);
    return kernel;
}




char* get_line(FILE* f) {
    int startAt = 0;
    char* buf = NULL;
    size_t size = 0;
    int INCREMENT = 80;
    
    /** The function terminates reading either after a new-line character
     is found or end-of-file is reached, or after (length - 1) characters have been read.
     If a new-line was reached it is included in the string as the last character
     before the null character. The length argument includes space needed
     for the null character which will be appended to the end of the string.
     As a result, to read N characters, the length specification must be specified as N+1.
     The string read is returned if at least one character was read and no error occurred,
     otherwise a NULL-pointer is returned.
     **/
    do {
        size += INCREMENT;
        buf = realloc(buf, size);
        fgets(buf+startAt, INCREMENT, f);
        startAt = strlen(buf); // Location of the '\0' at the end
        
    } while (!feof(f) && buf[startAt-1]!='\n');
    return buf;
}

char** readFile(FILE* f, size_t* nbLines) {
    *nbLines = 0;
    char** result = NULL;
    while(!feof(f)) {
        (*nbLines)++;
        result = realloc(result, (*nbLines) * sizeof(char*));
        result[(*nbLines)-1] = get_line(f);
    }
    return result;
}


char** readSourceCode(const char* fileName, size_t* nbLines) {
    FILE* file;
    file = fopen(fileName, "r");
    if (!file) {
        printf("FAILED TO OPEN SOURCE FILE %s\n", fileName);
        exit(100);
    }
    return readFile(file, nbLines);
}


void OCLInit() {
    // 1. Get a platform.
    cl_platform_id platform;
    cl_uint nb_platforms;
    cl_int res = clGetPlatformIDs( 1, &platform, &nb_platforms );
    if (res != CL_SUCCESS || nb_platforms == 0)
    printf("No OpenCL platform found");
    
    // 2. Find a gpu device.
    cl_device_id device;
    cl_uint nb_devices;
    res = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                         1,
                         &device,
                         &nb_devices);
    if (res != CL_SUCCESS || nb_devices == 0)
    printf("No OpenCL GPU devices found");
    
    // Some properties of the device
    unsigned long buffer;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buffer), &buffer, NULL);
    printf("  COMPUTE_UNITS = %d\n", buffer);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buffer), &buffer, NULL);
    printf("  CLOCK_FREQ = %d\n", buffer);
    clGetDeviceInfo(device,  CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buffer), &buffer, NULL);
    printf("  LOC MEM SIZE = %d\n", buffer);
    clGetDeviceInfo(device,  CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buffer), &buffer, NULL);
    printf("  GLOB MEM SIZE = %ld\n", buffer);
    char name[100];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char[100]), &name, NULL);
    printf("Selected device \"%s\"\n", name);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local), &max_local, NULL);
    printf("Max work group size:\"%d\"\n", max_local);
    
    // 3. Create a context and command queue on that device.
    context = clCreateContext( NULL,
                              1,
                              &device,
                              NULL, NULL, &res);
    if (res != CL_SUCCESS)
    printf("Could not create device context\n");
    queue = clCreateCommandQueue( context,
                                 device,
                                 0, &res );
    if (res != CL_SUCCESS)
    printf("Could not create command queue\n");
    
    size_t nbLines;
    char** source = readSourceCode(sourceFile, &nbLines);
    printf("Read %d line(s) of source code:\n", (int) nbLines);
    program = clCreateProgramWithSource( context,
                                        nbLines,
                                        source,
                                        NULL, &res );
    if (res != CL_SUCCESS) {
        printf("Kernel creation failed! Error: %d\n", res);
        exit(200);
    }
    
    res = clBuildProgram( program, 1, &device, "-I /usr/lib/", NULL, NULL );
    
    if (res != CL_SUCCESS) {
        printf("Kernel compilation failed! Error: %d\n", res);
        // Shows the log
        char* build_log;
        size_t log_size;
        // First call to know the proper size
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        build_log = malloc(sizeof(char)*(log_size+1));
        // Second call to get the log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        printf("%s\n", build_log);
        exit(200);
    }
}

void OCLRunKernelMult(cl_kernel kernel, int n) {
    // 6. Launch the kernel. Let OpenCL pick the local work size.
    size_t global_work_size = n/4; // <-- omwille van vector4 operaties
    int err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     1,      // dimensions
                                     NULL,   // must be NULL
                                     &global_work_size,
                                     NULL, 0, NULL, NULL);
    err_check(err, "executing kernel");
    clFinish( queue );
}

int OCLRunKernelSum(cl_mem* from, int n) {
    cl_int error;
    cl_kernel kernel = clCreateKernel(program, "sum", &error);
    err_check(error, "creating kernel");
    
    int half = n/2;
    size_t upperbound;
    if (max_local < half)
        upperbound = max_local;
    else
        upperbound = half;
    int iterations = (int) floor(log2((double) upperbound));
    size_t local_size = pow(2,iterations);
    
    printf("Work group size %d\n", local_size);
    int nb_groups = half / local_size;
    size_t global_work_size = nb_groups * local_size;
    
    clSetKernelArg(kernel, 0, sizeof(*from), from );
    clSetKernelArg(kernel, 1, sizeof(cl_int)*local_size, NULL );
    cl_mem oresult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                   sizeof(cl_int) * nb_groups, NULL, &error);
    clSetKernelArg(kernel, 2, sizeof(oresult), &oresult);
    
    cl_mem oit = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &iterations, &error);
    err_check(error, "creating nb. of iterations");
    clSetKernelArg(kernel, 3, sizeof(oit), &oit );
    
    printf("Running %d groups of %d items --> total is %d\n", nb_groups, local_size, global_work_size);
    int err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     1,      // dimensions
                                     NULL,   // must be NULL
                                     &global_work_size,
                                     &local_size,
                                     0, NULL, NULL);
    err_check(err, "executing kernel");
    clFinish( queue );
    cl_int* result = malloc(sizeof(cl_int)*nb_groups);
    err = clEnqueueReadBuffer (queue, oresult, CL_TRUE, 0, sizeof(cl_int)*nb_groups, result, 0, NULL, NULL);
    err_check(err, "reading GPU results (status value)");
    int acc = 0;
    for (int i = 0; i < nb_groups; i++) {
        printf("Adding %d\n", result[i]);
        acc += result[i];
    }
    return acc; // En de overschot??
}


int* OCLSync(int n, cl_mem buffer) {
    int* result = malloc(sizeof(int)*n);
    int err;
    err = clEnqueueReadBuffer (queue, buffer, CL_TRUE, 0, sizeof(cl_int)*n, result, 0, NULL, NULL);
    err_check(err, "reading GPU results (status value)");
    return result;
}

int main(int argc, const char * argv[]) {
    if (!argc) {
        printf("Run with argument n to compute dot product of vectors of size 2^n");
    }
    int e = atoi(argv[1]);
    int n = pow(2,e);
    int* vec1 = malloc(sizeof(int) * n);
    int* vec2 = malloc(sizeof(int) * n);
    int i;
    srand(0);
    for (i=0; i<n; i++) {
        vec1[i] = rand() % 500;
        vec2[i] = rand() % 500;
        vec1[i] = 2;
        vec2[i] = 3;
    }
    OCLInit();
    cl_mem oresult;
    printf("Preparing mult\n");
    cl_kernel kl = OCLPrepKernelMult(vec1, vec2, n, &oresult);
    int start = clock();
    OCLRunKernelMult(kl,n);
    clFinish( queue );
    printf("Preparing sum\n");
    cl_mem from = oresult;
    int sum = OCLRunKernelSum(&from, n);
    printf("Result: %d\n", sum);
    int end = clock();
    printf("Time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);
}
