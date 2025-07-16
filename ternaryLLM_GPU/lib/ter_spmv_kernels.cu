#include "ter_spmv_kernels.cuh"

__global__ void ter_spmv_baseline(float* X, int8_t* W1, float* result, int rows, int columns, int inners) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx >= columns)
        return;

    float res = 0;
    for (int i = 0; i < inners; i++) {
        res += X[i] * (float)W1[i * columns + tx];
    }
    result[tx] = res;
    // result[tx * rows + ty] = res;
}