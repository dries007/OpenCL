//
//  kernel.cl
//  Dotproduct
//
//  Created by Joost Vennekens on 11/02/15.
//  Copyright (c) 2015 Joost Vennekens. All rights reserved.
//

__kernel void multiply(__global int* v1,
                         __global int* v2,
                         __global int* res) {
    int i = get_global_id(0);
    res[i] = v1[i] * v2[i];
}


__kernel void multiply2(__global int* v1,
                       __global int* v2,
                       __global int* res) {
    int i = get_global_id(0);
    int4 mult= vload4(i,v1) * vload4(i,v2);
    vstore4(mult, i, res);
}


__kernel void sum(__global int* from,
                  __local int* vec,
                  __global int* result,
                  __global int* iterations) {
    int lid = get_local_id(0);
    int lid2 = lid * 2;
    int lsize = get_local_size(0);
    int g = get_group_id(0);
    int offset = g * lsize * 2;
    
    vec[lid] = from[offset+lid2] + from[offset+lid2+1];
    int n = *iterations;
    int k;
    int skip = 1;
    for (k=0; k<n; k++) {
        vec[lid2] = vec[lid2] + vec[lid2 + skip];
        skip *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        result[g] = vec[offset];
    }
}

//EOF