#include "ter_spmm_kernels.cuh"

__global__ void ter_spmm_float_baseline(float* X, int8_t* W1, float* result, int rows, int columns, int inners) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= columns || ty >= rows)
        return;

    float res = 0;
    for (int i = 0; i < inners; i++) {
        res += X[i * rows + ty] * (float)W1[i * columns + tx];
    }
    result[ty * columns + tx] = res;
    // result[tx * rows + ty] = res;
}


__global__ void ter_csc_spmm_float_baseline(float* X, 
                                            int32_t* w_neg_row_indice, int32_t* w_neg_col_offset, 
                                            int32_t* w_pos_row_indice, int32_t* w_pos_col_offset, 
                                            float* result, 
                                            int rows, int columns, int inners)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= columns || ty >= rows)
        return;
    
    float res = 0;

    int neg_start = w_neg_col_offset[tx];
    int pos_start = w_pos_col_offset[tx];

    int neg_end = w_neg_col_offset[tx + 1];
    int pos_end = w_pos_col_offset[tx + 1];

    int neg = neg_end - neg_start;
    int pos = pos_end - pos_start;

    int min_neg_pos = neg < pos ? neg : pos;

    for (int k = 0; k < min_neg_pos; k++) {
        res -= X[w_neg_row_indice[neg_start + k] * rows + ty];
        res += X[w_pos_row_indice[pos_start + k] * rows + ty];
    }
    for (int k = min_neg_pos; k < pos; k++) {
        res += X[w_pos_row_indice[pos_start + k] * rows + ty];
    }
    for (int k = min_neg_pos; k < neg; k++) {
        res -= X[w_neg_row_indice[neg_start + k] * rows + ty];
    }

    result[ty * columns + tx] = res;
    // result[tx * rows + ty] = res;
}

__global__ void ter_mcsc_spmm_float_baseline(const float* __restrict__ X, 
                                            const int32_t* __restrict__ w_merged_row_indice, 
                                            const int32_t* __restrict__ w_merged_col_offset, 
                                            float* result, 
                                            int rows, int columns, int inners)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= columns || row >= rows)
        return;

    float res = 0;
    const int4* w_merged_col_offset_int4 = reinterpret_cast<const int4*>(w_merged_col_offset);
    int4 col_offset = __ldg(w_merged_col_offset_int4 + col);

    // printf("%d:%d:%d:%d\n", col_offset.x, col_offset.y, col_offset.z, col_offset.w);

    for (int i = col_offset.x; i < col_offset.y; i+=2) {
        res += X[w_merged_row_indice[i]*rows + row];
        res -= X[w_merged_row_indice[i+1]*rows + row];
    }

    float partial_sum = 0;
    for (int i = col_offset.y; i < col_offset.z; i++) {
        partial_sum += X[w_merged_row_indice[i]*rows + row];
    }

    res += col_offset.w*partial_sum;
    result[row * columns + col] = res;
}