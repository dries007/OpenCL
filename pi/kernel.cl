
__kernel void estimatepi(unsigned int n, __global float * inp, __global unsigned int * result)
{
    int i = get_global_id(0);

    if (i >= n) return;

    i *= 2;

    result[i] = (inp[i] * inp[i] + inp[i+1] * inp[i+1]) <= 1.0f ? 1 : 0;

    return;
}

