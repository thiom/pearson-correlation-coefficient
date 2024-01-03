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

Since the correlations are symmetric, the function will only compute the upper triangle of the result matrix.

- [CPU only, not vectorized, double precision](./double-prec/)

- [CPU only, avx-512, double precision](./avx512-double-prec/)

- [CPU only, avx-512, single precision](./avx512-single-prec/)

- [GPU with cuda, single precision](./cuda-single-prec/)
