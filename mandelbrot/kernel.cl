#define MIN_REAL -2.0
#define MAX_REAL 1.0
#define MIN_IMAGINARY -1.1f
#define MAX_IMAGINARY (MIN_IMAGINARY + (MAX_REAL - MIN_REAL) * ((float)HEIGHT) / ((float)WIDTH))
//#define MAX_IMAGINARY 1.2
#define MAX_ITERATIONS 200

#define IMAGINARY_POS(y) (float)(MAX_IMAGINARY - (y) * ((MAX_IMAGINARY - MIN_IMAGINARY) / (float)((HEIGHT) - 1)))

#define REAL_POS(x) (float)(MIN_REAL + (x) * ((MAX_REAL - MIN_REAL) / (float)((WIDTH) - 1)))

float calc_mandel_pixel(int x_pos, int y_pos, int WIDTH, int HEIGHT)
{
    float real = REAL_POS(x_pos);
    float img = IMAGINARY_POS(y_pos);

    float z_real = real;
    float z_img = img;

    //bool is_inside = true;

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        float z_real_squared = z_real * z_real;
        float z_img_squared = z_img * z_img;

        if (z_real_squared + z_img_squared > 4)
        {
            // Not in the mandelbrot set
            return ((float) i / (float)MAX_ITERATIONS) * 255.f;
        }
        float tmp = z_real_squared - z_img_squared + real;
        z_img = 2.f * z_real * z_img + img;
        z_real = tmp;
    }
    return 0.f;
}

__kernel void mandelbrot(int w, int h, __global float *result)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= w || y >= h)
        return;

    result[y * w + x] = calc_mandel_pixel(x, y, w, h);
    return;
}

