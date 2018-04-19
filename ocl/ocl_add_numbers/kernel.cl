__kernel void add_numbers(__global float *vector_a,
                          __global float *vector_b,
                          int size,
                          __global float *result)
{
    const int gid = get_global_id(0);

    if (gid >= size)
        return;

    result[gid] = vector_a[gid] + vector_b[gid];
}

