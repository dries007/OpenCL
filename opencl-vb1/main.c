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
#include <stdlib.h>
#include <string.h>


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
char* get_line(FILE* f);

const char* sourceFile = "./kernel.cl";

cl_context context;
cl_command_queue queue;
cl_program program;

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
        printf("%s", result[(*nbLines)-1]);
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
    unsigned int buffer;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buffer), &buffer, NULL);
    printf("  COMPUTE_UNITS = %d\n", buffer);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buffer), &buffer, NULL);
    printf("  CLOCK_FREQ = %d\n", buffer);
    char name[100];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char[100]), &name, NULL);
    printf("Selected device \"%s\"\n", name);
    
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

void err_check(int error, char* loc) {
    if (error != CL_SUCCESS) {
        printf("ERROR: %d when %s\n", error, loc);
        exit(300);
    }
}

cl_kernel OCLPrepKernelMult(int* vec1, int* vec2, int n, cl_mem* result, cl_mem* ovec_out) {
    cl_int error;
    cl_kernel kernel = clCreateKernel(program, "multiply", &error );
    err_check(error, "creating kernel");
    
    cl_mem ovec1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_int) * n, vec1, &error);
    err_check(error, "creating buffer for first vector");
    clSetKernelArg(kernel, 0, sizeof(ovec1), &ovec1);
    
    cl_mem ovec2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_int) * n, vec2, &error);
    err_check(error, "creating buffer for second vector");
    clSetKernelArg(kernel, 1, sizeof(ovec2), &ovec2);
    
    *result = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(cl_int) * n, NULL, &error);
    err_check(error, "creating buffer for result");
    clSetKernelArg(kernel, 2, sizeof(*result), result);
    
    *ovec_out = ovec1;
    return kernel;
}

void OCLRunKernelMult(cl_kernel kernel, int n) {
    // 6. Launch the kernel. Let OpenCL pick the local work size.
    size_t global_work_size = n;
    int err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     1,      // dimensions
                                     NULL,   // must be NULL
                                     &global_work_size,
                                     NULL,   // local work size
                                     0, NULL,// wait-for list
                                     NULL    // event
                                     );
    err_check(err, "executing kernel");
    clFinish( queue );
}

cl_kernel OCLPrepKernelSum(cl_mem* from, cl_mem* to) {
    cl_int error;
    cl_kernel kernel = clCreateKernel(program, "sum", &error );
    err_check(error, "creating kernel");
    clSetKernelArg(kernel, 0, sizeof(*from), from );
    clSetKernelArg(kernel, 1, sizeof(*to), to);
    return kernel;
}


void OCLRunKernelSum(cl_kernel kernel, int n) {
    // 6. Launch the kernel. Let OpenCL pick the local work size.
    size_t global_work_size = n/2;
    int err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     1,      // dimensions
                                     NULL,   // must be NULL
                                     &global_work_size,
                                     NULL, 0, NULL, NULL);
    err_check(err, "executing kernel");
    clFinish( queue );
}


int* OCLSync(int n, cl_mem buffer) {
    int* result = malloc(sizeof(int)*n);
    int err;
    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(cl_int)*n, result, 0, NULL, NULL);
    err_check(err, "reading GPU results (status value)");
    return result;
}   

int main(int argc, const char * argv[]) {
    if (!argc) {
        printf("Run with argument n to compute dot product of vectors of size 2^n");
    }
    int e = atoi(argv[1]);
    size_t n = pow(2, e);

    int* vec1 = malloc(sizeof(int) * n);
    int* vec2 = malloc(sizeof(int) * n);
    if (vec1 == NULL || vec2 == NULL) {
        printf("malloc failed. Out of RAM? n=%d\n", 2 * n * sizeof(int));
        abort();
    }
    int i;
    for (i=0; i<n; i++) {
        vec1[i] = 2;
        vec2[i] = 3;
    }
    int start = clock();
    cl_mem oresult;
    OCLInit();
    cl_mem ovec1;
    cl_kernel kl = OCLPrepKernelMult(vec1, vec2, n, &oresult, &ovec1);
    OCLRunKernelMult(kl,n);
    int* result = OCLSync(n, oresult);
    clFinish( queue );
    #ifdef DEBUGX
    for (i=0; i<n; i++)
        printf("%d --> %d\n", i, result[i]);
    #endif
    int old_size = n;
    int new_size;
    cl_mem to = ovec1;
    cl_mem from = oresult;
    cl_mem tmp;
    int* half;
    while (old_size > 1){
        new_size = old_size / 2;
        half = malloc(new_size * sizeof(int));
        kl = OCLPrepKernelSum(&from, &to);
        OCLRunKernelSum(kl, old_size);
        old_size = new_size;
        result = half;
        tmp = to;
        to = from;
        from = tmp;
        result = OCLSync(1,from);
        printf("Result: %d\n",result[0]);
    }
    result = OCLSync(1,from);
    int end = clock();
    printf("Time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);
}

#pragma clang diagnostic pop