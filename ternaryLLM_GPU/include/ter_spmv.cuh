#pragma once
#include "ter_spmv_kernels.cuh"
#include "util.h"
#include <chrono>

class Ternary_SpMV {
public:
    enum KernelType {
        BASELINE,
        TER_MCSC_VEC
    };
public:
    Ternary_SpMV() = default;

    __host__ void ter_spmv_cusparse_spmv(TerSparseDataWrap<float>& spmv, SpMMStat& stat);

    __host__ void ter_spmv_cublas_spmv(TerSparseDataWrap<float>& spmv, SpMMStat& stat);

    template<typename D=float> 
    __host__ void ter_spmv_wrapper(TerSparseDataWrap<D>& spmv, SpMMStat& stat, KernelType ktype)
    {
        CUDA_CALL_CHECK(cudaSetDevice(0));
        CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */

        auto fn_s = std::chrono::high_resolution_clock::now();

        /* ter csc device pointers */
        int* dev_w1_neg_row_indices = 0;
        int* dev_w1_pos_row_indices = 0;

        int* dev_w1_neg_col_offset = 0;
        int* dev_w1_pos_col_offset = 0;

        int* dev_w1_neg_col_offset_tiled = 0;
        int* dev_w1_pos_col_offset_tiled = 0;

        int* dev_w1_merged_col_offset = 0;
        int* dev_w1_merged_row_indice = 0;

        int* dev_w1_merged_col_offset_tiled = 0;
        int* dev_w1_merged_row_indice_tiled = 0;

        auto prekn_s = std::chrono::high_resolution_clock::now();

        if (ktype == BASELINE) {
            /* allocate device memory for csc */
            CUDA_CALL_CHECK(cudaMalloc((void**)(&spmv.dev_w1), spmv.size_w1*sizeof(int8_t)));

            /* copy to device */
            CUDA_CALL_CHECK(cudaMemcpy((void*)spmv.dev_w1, (void*)spmv.host_w1.data(), spmv.size_w1*sizeof(int8_t), cudaMemcpyHostToDevice));
        }

        if (ktype == TER_MCSC_VEC) {
            /* allocate device memory for csc */
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_merged_row_indice), spmv.w1_cnt_nnz*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_merged_col_offset), spmv.size_csc_col_offset_merged*sizeof(int)));

            /* copy to device */
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_merged_row_indice, (void*)spmv.w1_merged_row_indice, spmv.w1_cnt_nnz*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_merged_col_offset, (void*)spmv.w1_merged_col_offset, spmv.size_csc_col_offset_merged*sizeof(int), cudaMemcpyHostToDevice));
        
            stat.fn_mem_use[spmv.sparsity][stat.curr_config ] = spmv.w1_cnt_nnz*sizeof(int) + spmv.size_csc_col_offset_merged*sizeof(int);
        }

        /* allocate device memory for matrix */
        CUDA_CALL_CHECK(cudaMalloc((void**)(&spmv.dev_x), spmv.size_x*sizeof(float)));
        CUDA_CALL_CHECK(cudaMalloc((void**)(&spmv.dev_res), spmv.size_res*sizeof(float)));
        CUDA_CALL_CHECK(cudaMemcpy((void*)spmv.dev_x, (void*)spmv.host_x.data(), spmv.size_x*sizeof(float), cudaMemcpyHostToDevice));
        
        auto prekn_e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> prekn_span = prekn_e - prekn_s; // data prepare duration

        /* call cuda kernel */
        auto kernel_config = config_ter_spmv_kernel(spmv.rows, spmv.columns, spmv.inners, ktype);
        cudaEvent_t start, stop;
        float kn_span = 0;  // ms
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        switch (ktype)
        {
        case BASELINE:
            INFO(spmv.rows << "::");
            ter_spmv_baseline<<<kernel_config.first, kernel_config.second>>>(spmv.dev_x, spmv.dev_w1, spmv.dev_res, 
                                                                                spmv.rows, spmv.columns, spmv.inners);
            break;
        case TER_MCSC_VEC:
            ter_cscvec_spmv<1><<<kernel_config.first, kernel_config.second>>>(spmv.rows, spmv.columns, spmv.inners,
                                                                                spmv.dev_x, 
                                                                                dev_w1_merged_row_indice,
                                                                                dev_w1_merged_col_offset,
                                                                                spmv.dev_res);
            break;
        default:
            break;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kn_span, start, stop);
        stat.kn_spans[stat.curr_config][spmv.sparsity].push_back(kn_span);

        auto postkn_s = std::chrono::high_resolution_clock::now();

        CUDA_CALL_CHECK(cudaMemcpy((void*)spmv.host_res.data(), (void*)spmv.dev_res, spmv.size_res*sizeof(float), cudaMemcpyDeviceToHost));

        /* free */
        CUDA_CALL_CHECK(cudaFree(spmv.dev_x));
        CUDA_CALL_CHECK(cudaFree(spmv.dev_res));

        if (ktype == BASELINE) {
            CUDA_CALL_CHECK(cudaFree(spmv.dev_w1));
        }

        if (ktype == TER_MCSC_VEC) {
            CUDA_CALL_CHECK(cudaFree(dev_w1_merged_row_indice));
            CUDA_CALL_CHECK(cudaFree(dev_w1_merged_col_offset));
        }

        auto fn_e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> postkn_span = fn_e - postkn_s;
        std::chrono::duration<double, std::milli> fn_span = fn_e - fn_s;
        stat.fn_spans[stat.curr_config][spmv.sparsity].push_back(fn_span.count());

        INFO(spmv.sparsity << " Memory: " << stat.fn_mem_use[stat.curr_config][spmv.sparsity]/1024 << "KB" << " Kernel: " << kn_span << "ms" << " Pre: " << prekn_span.count() << "ms" << " Post: " << postkn_span.count() << "ms" << " Function: " << fn_span.count() << "ms");
    }

private:
std::pair<dim3, dim3> config_ter_spmv_kernel(const int M, const int N, const int K, KernelType ktype) {
    std::pair<dim3, dim3> config;
    switch (ktype)
    {
    case BASELINE:
    {
        int blkx = 32;
        int blky = 1;
        int gridx = (int)ceil(1.*N/blkx);
        int gridy = 1;
        config.first = dim3(gridx, gridy);
        config.second = dim3(blkx, blky);
        break;
    }
    case TER_MCSC_VEC:
    {
        int blkx = 512;                           // Tile Width in N-dim
        int gridx = (int)ceil(1.*32*N/blkx);           // Number of Tile in N-dim: N/Tile_Width
        config.first = dim3(gridx);
        config.second = dim3(blkx);
        break;
    }
    default:
        config.first = dim3(32, 32);
        config.second = dim3(32, 32);
        break;
    }

    return config;
    }
};