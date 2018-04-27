/**
 * Dries Kennes
 */

__kernel void estimatepi(unsigned int n, unsigned int group_size, __global float * inpx, __global float * inpy, __global unsigned int * result)
{
    int group = get_global_id(0);

    if (group >= n) return;

    int count = 0;
    for (int i = group_size * group; i < group_size + (group_size * group); i++)
    {
        count += ((inpx[i] * inpx[i]) + (inpy[i] * inpy[i])) < 1.0f;
    }
    result[group] = count;

    return;
}
