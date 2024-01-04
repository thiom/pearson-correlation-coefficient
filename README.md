## Pearson correlation coefficient

This is a collection of a few different parallel algorithms for calculating the 
[pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

The function:

```
void correlate(int ny, int nx, const float* data, float* result)
```

Given by the input matrix **data**, with **ny** rows and **nx** columns, where **0 <= y < ny** and **0 <= x < nx**, 
the element at row **y** and column **x** is stored in **data[x + y \* nx]**.

For all **i** and **j** with **0 <= j <= i < ny** the function calculates the correlation coefficients between all rows **i** 
and **j**, and stores the results in **result[i + j \* ny]**.

Since the correlations are symmetric, the function will only compute the upper triangle of the result matrix. Here are the 
different versions and some benchmark results with various input sizes (in pixels):

- [CPU only, not vectorized, double precision](./double-prec/)  
4000 x 1000: 0.46 s

- [CPU only, avx-512, double precision](./avx512-double-prec/)  
4000 x 1000: 0.084 s  
6000 x 6000: 1.13 s  
9000 x 9000: 3.47 s  

- [CPU only, avx-512, single precision](./avx512-single-prec/)  
4000 x 1000: 0.041 s  
6000 x 6000: 0.37 s  
9000 x 9000: 1.36 s  

- [GPU with cuda, single precision](./cuda-single-prec/)  
4000 x 1000: 0.095 s  
6000 x 6000: 0.55 s  
9000 x 9000: 2.64 s  

The CPU versions perform pretty decently, but the cuda implementation could be improved significantly. The benchmarks were 
done on a machine with following specs:

- Intel Xeon W-2255, 10 cores, 20 threads, 3.70 / 4.50 GHz
- 64GB DDR4 (4 x 16)
- Nvidia Quadro RTX 4000
