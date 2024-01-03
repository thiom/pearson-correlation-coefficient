#include <cmath>
#include <vector>
#include <x86intrin.h>

inline void* aligned_malloc(std::size_t bytes) {
    void* ret = nullptr;
    if (posix_memalign(&ret, 32, bytes)) {
        return nullptr;
    }
    return ret;
}

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

inline double4_t* double4_alloc(std::size_t n) {
    return static_cast<double4_t*>(aligned_malloc(sizeof(double4_t) * n));
}
static inline double4_t swap2(double4_t x) {
    return _mm256_permute2f128_pd(x, x, 0b00000001);
}
static inline double4_t swap1(double4_t x) {
    return _mm256_permute_pd(x, 0b00000101);
}

double row_mean(const float* row, int nx){
    double sum = 0.0, mean;
    for (int j=0; j < nx; j++){
        sum += (double) row[j];
    }
    mean = sum / (double) nx;
    return mean;
}


double row_root_sq_sum(const float* row, int nx, double mean){
    double square_sum = 0.0, s;
    for(int j=0; j < nx; j++){
        double val = (double) row[j];
        square_sum += ((val - mean) * (val - mean));
    }
    s = sqrt(square_sum);
    return s;
}

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> nd(ny*nx);

    #pragma omp parallel for schedule(static, 1)
    for (int j=0; j < ny; j++){
        double r_mean = row_mean(&data[j * nx], nx);
        double row_rss = row_root_sq_sum(&data[j * nx], nx, r_mean);
        for(int i=0; i < nx; i++){
            nd[i + j * nx] = (((double) data[i + j * nx]) - r_mean) / row_rss;
        }
    }
    //number of elements per vector
    const int nb = 4;
    //number of vectors per row
    int na = (ny + nb - 1) / nb;
    double4_t* vd = double4_alloc(na * nx);

    //vectorization
    #pragma omp parallel for schedule(static, 1)
    for(int ja=0; ja < na; ja++){
        for(int i=0; i < nx; i++){
            for(int jb=0; jb < nb; jb++){
                const int PF = 20;
                int j = nb * ja + jb;
                __builtin_prefetch(&nd[nx * j + i + PF]);
                vd[nx * ja + i][jb] = j < ny ? nd[nx * j + i] : 0.0;
            }
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for(int ia=0; ia < na; ia++){
        for(int ja = ia; ja < na; ja++){
            double4_t z00 = {0,0,0,0};
            double4_t z01 = {0,0,0,0};
            double4_t z10 = {0,0,0,0};
            double4_t z11 = {0,0,0,0};
            for(int k=0; k < nx; k++){
                const int PF = 12;
                __builtin_prefetch(&vd[nx * ia + k + PF]);
                __builtin_prefetch(&vd[nx * ja + k + PF]);

                double4_t a00 = vd[nx * ia + k];
                double4_t b00 = vd[nx * ja + k];
                double4_t a10 = swap2(a00);
                double4_t b01 = swap1(b00);

                z00 = z00 + (a00 * b00);
                z01 = z01 + (a00 * b01);
                z10 = z10 + (a10 * b00);
                z11 = z11 + (a10 * b01);
            }
            double4_t z[4] = {z00, z01, z10, z11};
            for(int kb=1; kb < nb; kb += 2){
                z[kb] = swap1(z[kb]);
            }
            for(int jb=0; jb < nb; jb++){
                for(int ib=0; ib < nb; ib++){
                    int i = ib + nb * ia;
                    int j = jb + nb * ja;
                    if(j < ny && i < ny && i <= j){
                        result[ny * i + j] = z[ib ^ jb][jb];
                    }
                }
            }
        }
    }
    free(vd);
}
