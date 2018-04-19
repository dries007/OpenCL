cmake_minimum_required(VERSION 3.9)
project(OpenCL C)

set(CMAKE_C_STANDARD 99)

FIND_LIBRARY(OPENCL_LIBRARIES OpenCL
        PATHS ENV LD_LIBRARY_PATH ENV OpenCL_LIBPATH
)

GET_FILENAME_COMPONENT(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)
GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})
message("OpenCL: Incl: ${OPENCL_INCLUDE_DIR} Libs: ${OPENCL_LIBRARIES}")


add_executable(OpenCL main.c)
target_link_libraries(OpenCL ${OPENCL_LIBRARIES} m)

add_executable(OpenCL_vb1 opencl-vb1/main.c)
target_link_libraries(OpenCL_vb1 ${OPENCL_LIBRARIES} m)

add_executable(OpenCL_vb2 opencl-vb2/main.c)
target_link_libraries(OpenCL_vb2 ${OPENCL_LIBRARIES} m)

add_executable(Platforms showall/platforms.c)
target_link_libraries(Platforms ${OPENCL_LIBRARIES} m)

add_executable(Devices showall/devices.c)
target_link_libraries(Devices ${OPENCL_LIBRARIES} m)

set(OCL_LIB lib/ocl_utils.h lib/ocl_utils.c lib/time_utils.h lib/time_utils.c)
include_directories(lib)

add_executable(Mandelbrot mandelbrot/main.c ${OCL_LIB} mandelbrot/image_helper.c mandelbrot/image_helper.h)
target_link_libraries(Mandelbrot ${OPENCL_LIBRARIES} m png)

add_executable(Pi pi/main.c ${OCL_LIB})
target_link_libraries(Pi ${OPENCL_LIBRARIES} m)
