//
// Created by dries on 30/03/18.
//

#ifndef OPENCL_IMAGE_HELPER_H
#define OPENCL_IMAGE_HELPER_H

extern void setRGB(unsigned char *ptr, float val);

int writeImage(char* filename, int width, int height, float *buffer, char* title);

#endif //OPENCL_IMAGE_HELPER_H
