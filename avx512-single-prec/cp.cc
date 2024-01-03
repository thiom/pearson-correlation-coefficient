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

typedef float float8_t __attribute__ ((vector_size (32)));
const float8_t float8_0 = {0,0,0,0,0,0,0,0};
inline float8_t* float8_alloc(std::size_t n) {
    return static_cast<float8_t*>(aligned_malloc(sizeof(float8_t) * n));
}

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

float row_mean(const float* row, int nx){
    float sum = 0.0, mean;
    for (int j=0; j < nx; j++){
        sum += (float) row[j];
    }
    mean = sum / (float) nx;
    return mean;
}


float row_root_sq_sum(const float* row, int nx, float mean){
    float square_sum = 0.0, s;
    for(int j=0; j < nx; j++){
        float val = (float) row[j];
        square_sum += ((val - mean) * (val - mean));
    }
    s = sqrt(square_sum);
    return s;
}


void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<float> nd(ny*nx);

    #pragma omp parallel for schedule(static, 1)
    for (int j=0; j < ny; j++){
        float r_mean = row_mean(&data[j * nx], nx);
        float row_rss = row_root_sq_sum(&data[j * nx], nx, r_mean);
        for(int i=0; i < nx; i++){
            nd[i + j * nx] = (((float) data[i + j * nx]) - r_mean) / row_rss;
        }
    }
    //number of elements per vector
    const int nb = 8;
    //number of vectors per row
    int na = (ny + nb - 1) / nb;
    float8_t* vd = float8_alloc(na * nx);

    //vectorization
    #pragma omp parallel for schedule(static, 1)
    for(int ja=0; ja < na; ja++){
        for(int i=0; i < nx; i++){
            for(int jb=0; jb < nb; jb++){
                int j = ja * nb + jb;
                vd[nx * ja + i][jb] = j < ny ? nd[nx * j + i] : 0.0;
            }
        }
    }
    #pragma omp parallel for schedule(static, 1)
    for(int ia=0; ia < na; ia++){
        for(int ja = ia; ja < na; ja++){
            float8_t z000 = float8_0;
            float8_t z001 = float8_0;
            float8_t z010 = float8_0;
            float8_t z011 = float8_0;
            float8_t z100 = float8_0;
            float8_t z101 = float8_0;
            float8_t z110 = float8_0;
            float8_t z111 = float8_0;
            for(int k=0; k < nx; k++){
                const int PF = 20;
                __builtin_prefetch(&vd[nx * ia + k + PF]);
                __builtin_prefetch(&vd[nx * ja + k + PF]);

                float8_t a000 = vd[nx * ia + k];
                float8_t b000 = vd[nx * ja + k];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);

                z000 = z000 + (a000 * b000);
                z001 = z001 + (a000 * b001);
                z010 = z010 + (a010 * b000);
                z011 = z011 + (a010 * b001);
                z100 = z100 + (a100 * b000);
                z101 = z101 + (a100 * b001);
                z110 = z110 + (a110 * b000);
                z111 = z111 + (a110 * b001);
                
            }
            float8_t z[8] = {z000, z001, z010, z011, z100, z101, z110, z111};
            for(int kb=1; kb < nb; kb += 2){
                z[kb] = swap1(z[kb]);
            }
            for(int jb=0; jb < nb; jb++){
                for(int ib=0; ib < nb; ib++){
                    int i = ib + ia * nb;
                    int j = jb + ja * nb;
                    if(j < ny && i < ny && i <= j){
                        result[ny * i + j] = z[ib ^ jb][jb];
                    }
                }
            }
        }
    }
    free(vd);
}
