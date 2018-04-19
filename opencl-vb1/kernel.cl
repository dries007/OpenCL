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

__kernel void sum(__global int* v,
                  __global int* res) {
    int i = get_global_id(0);
    int j = i * 2;
    res[i] = v[j] + v[j+1];
}

// EOF