#pragma once

#include <cuda.h>
#include <assert.h>
#include <cusparse.h>
#include <cublas_v2.h>
// #include <cusparseLt.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define CUDA_CALL_CHECK(func_call)              \
    do {                                        \
        const cudaError_t cudaerr = func_call;  \
        assert(cudaerr == cudaSuccess && __FILE__ && __LINE__);         \
    } while(0)

#define CUSPARSE_CALL_CHECK(func_call)                                                   \
{                                                                              \
    cusparseStatus_t status = (func_call);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        assert(false && __FILE__ && __LINE__);                \
    }                                                                          \
}

#define CUBLAS_CALL_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)



__global__ void ter_spmv_baseline(float* X, int8_t* W1, float* result, int rows, int columns, int inners);


#define FULL_WARP_MASK 0xFFFFFFFF
template <class T>
/**
 *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
 *  the value of the val variable from the thread at lane X+offset of the same warp.
 *  The data exchange is performed between registers, and more efficient than going
 *  through shared memory, which requires a load, a store and an extra register to
 *  hold the address.
 */
__device__ void warp_reduce_sum(T& val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync (FULL_WARP_MASK, val, offset);
}


template<int TILE, typename DATA_TYPE=float>
/**
 * @brief 
 * 
 * @tparam TILE 
 * @tparam DATA_TYPE 
 * @param x_vec 
 * @param w_merged_row_indice 
 * @param w_merged_col_offset 
 * @param result 
 */
__global__ void ter_cscvec_spmv(
    const unsigned int rows, 
    const unsigned int columns, 
    const unsigned int inners,
    const DATA_TYPE* __restrict__ x_vec, 
    const int32_t* __restrict__ w_merged_row_indice, 
    const int32_t* __restrict__ w_merged_col_offset,
    DATA_TYPE* result
)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;   // thread id in a grid
    const unsigned int warp_id = thread_id / 32;    // logical warp id
    const unsigned int lane_id = thread_id % 32;    // thread id in a warp

    const unsigned int col = warp_id; ///< One warp per column

    DATA_TYPE sum = 0;
    if (col < columns) {
        const int4* w_tiled_merged_col_offset_int4 = reinterpret_cast<const int4*>(w_merged_col_offset + col*4);
        int4 col_offset = __ldca(w_tiled_merged_col_offset_int4);

        for (int i = col_offset.x + lane_id*2; i < col_offset.y; i+=2*32) {
            int pos_id = __ldca(w_merged_row_indice + i);
            int neg_id = __ldca(w_merged_row_indice + i + 1);
            sum += x_vec[pos_id] - x_vec[neg_id];
        }

        DATA_TYPE partial_sum = 0;
        for (int i = col_offset.y + lane_id; i < col_offset.z; i+=32) {
            int common_id = __ldca(w_merged_row_indice + i);
            partial_sum += x_vec[common_id];
        }

        sum += col_offset.w*partial_sum;
    }

    warp_reduce_sum(sum);

    if (lane_id == 0 && col < columns) {
        result[col] = sum;
    }
}


template<typename DATA_TYPE=float>
__global__ void ter_cscadp_spmv(
    const unsigned int rows, 
    const unsigned int columns, 
    const unsigned int inners,
    const DATA_TYPE* __restrict__ x_vec, 
    const int32_t* __restrict__ w_merged_row_indice, 
    const int32_t* __restrict__ w_merged_col_offset,
    const int32_t* __restrict__ w_merged_col_blk,
    DATA_TYPE* result
)
{
    return;
}