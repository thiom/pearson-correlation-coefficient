#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "math.h"

static inline void check(cudaError_t err, const char* context) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}
static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

__global__ void normalization_kernel(int ny, int nx, float* data, float* ntdata) {
    int y = blockIdx.x;
    if(y >= ny) return;

    float s = 0.0;
    for(int x=0; x < nx; x++) {
        float v = data[x+y*nx];
        s += v;
    }
    float m = s / (float) nx;
    float rs = 0.0;

    for(int x=0; x < nx; x++) {
        float v = data[x + y * nx];
        rs += ((v - m) * (v - m));
    }
    float r = sqrt(rs);
    for(int x=0; x < nx; x++) {
        float v = ((data[x + y * nx]) - m ) / r;
        ntdata[y + x * ny] = v;
    }
}

__global__ void matmul_kernel(int ny, int nx, float* ntdata, float* r){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= ny || j >= ny) return;
    float s = 0.0;
    if(i <= j) {
        for(int k=0; k < nx; k++) {
            float x = ntdata[ny * k + j];
            float y = ntdata[ny * k + i];
            s += (x * y);
        }
    }
    r[j + i * ny] = s;
}

void correlate(int ny, int nx, const float* data, float* result) {
    int n = ny * nx * sizeof(float);
    int rn = ny * ny * sizeof(float);

    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, n));

    float* ntGPU = NULL;
    CHECK(cudaMalloc((void**)&ntGPU, n));

    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, rn));

    CHECK(cudaMemcpy(dGPU, data, n, cudaMemcpyHostToDevice));

    int nBlocks = roundup(ny, 64);
    {
        normalization_kernel<<<nBlocks, 1>>>(ny, nx, dGPU, ntGPU);
        CHECK(cudaGetLastError());
    }
    {
        dim3 dimBlock(16, 16);
        dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
        matmul_kernel<<<dimGrid, dimBlock>>>(ny, nx, ntGPU, rGPU);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaMemcpy(result, rGPU, rn, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(ntGPU));
    CHECK(cudaFree(rGPU));
}
