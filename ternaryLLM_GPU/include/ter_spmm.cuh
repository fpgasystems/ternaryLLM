#pragma once
#include "ter_spmm_kernels.cuh"
#include "util.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <chrono>

class Ternary_SpMM {
public:
    enum SpMMType {
        FP32,
        INT8,
        INT32
    };

    enum KernelType {
        BASELINE,
        TER_CSC_BASE,
        TER_CSC_OPT,
        TER_TM_CSC,
        TER_TM_CSC_T,
        TER_M_CSC
    };

public:
    Ternary_SpMM() = default;

    __host__ void ter_spmm_cusparse_spmm(TerSparseDataWrap<float>& spmm, SpMMStat& stat);

    __host__ void ter_spmm_cublas_spmm(TerSparseDataWrap<float>& spmm, SpMMStat& stat);

    /**
     * Ternary SpMM Wrapper 
     * TILE_M: tile width in M-dim
     * TILE_N: tile width in N-dim
     * TILE_K: tile width in K-dim
     * F: vectorized data type
     */
    template<int TILE_M, int TILE_N, int TILE_K, typename F, typename D=float> 
    __host__ void ter_spmm_wrapper(TerSparseDataWrap<float>& spmm, SpMMStat& stat, KernelType ktype) {
        // CUDA_CALL_CHECK(cudaSetDevice(0));
        // CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */

        auto fn_s = std::chrono::high_resolution_clock::now();

        /* ter csc device pointers */
        int* dev_w1_neg_row_indices = 0;
        int* dev_w1_pos_row_indices = 0;

        int* dev_w1_neg_col_offset = 0;
        int* dev_w1_pos_col_offset = 0;

        int* dev_w1_neg_col_offset_tiled = 0;
        int* dev_w1_pos_col_offset_tiled = 0;

        int* dev_w1_swizzled_col_indices = 0;
        int* dev_w1_merged_col_offset = 0;
        int* dev_w1_merged_row_indice = 0;

        int* dev_w1_merged_col_offset_tiled = 0;
        int* dev_w1_merged_row_indice_tiled = 0;
        int* dev_w1_col_tile_indices = 0;

        auto prekn_s = std::chrono::high_resolution_clock::now();
        
        /* transfer csc pointers when necessary */
        if (ktype == TER_CSC_BASE) {
            //INFO("Benchmark ternary CSC Baseline");
            /* allocate device memory for csc */
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_neg_row_indices), spmm.w1_cnt_neg*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_neg_col_offset), (spmm.columns+1)*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_pos_row_indices), spmm.w1_cnt_pos*sizeof(float)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_pos_col_offset), (spmm.columns+1)*sizeof(int)));
            
            /* copy to device */
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_neg_row_indices, (void*)spmm.w1_neg_row_indice, spmm.w1_cnt_neg*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_neg_col_offset, (void*)spmm.w1_neg_col_offset, (spmm.columns+1)*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_pos_row_indices, (void*)spmm.w1_pos_row_indice, spmm.w1_cnt_pos*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_pos_col_offset, (void*)spmm.w1_pos_col_offset, (spmm.columns+1)*sizeof(int), cudaMemcpyHostToDevice));
        }

        if (ktype == TER_CSC_OPT) {
            //INFO("Benchmark ternary CSC Optimized");
            /* allocate device memory for csc */
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_neg_row_indices), spmm.w1_cnt_neg*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_neg_col_offset_tiled), spmm.size_csc_col_offset_tiled*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_pos_row_indices), spmm.w1_cnt_pos*sizeof(float)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_pos_col_offset_tiled), spmm.size_csc_col_offset_tiled*sizeof(int)));

            /* copy to device */
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_neg_row_indices, (void*)spmm.w1_neg_row_indice, spmm.w1_cnt_neg*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_neg_col_offset_tiled, (void*)spmm.w1_neg_col_offset_tiled, spmm.size_csc_col_offset_tiled*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_pos_row_indices, (void*)spmm.w1_pos_row_indice, spmm.w1_cnt_pos*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_pos_col_offset_tiled, (void*)spmm.w1_pos_col_offset_tiled, spmm.size_csc_col_offset_tiled*sizeof(int), cudaMemcpyHostToDevice));
        }

        if (ktype == TER_TM_CSC) {
            //INFO("Benchmark ternary CSC Optimized Tiled Merged");
            /* allocate device memory for csc */
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_merged_row_indice_tiled), spmm.w1_cnt_nnz*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_merged_col_offset_tiled), spmm.size_mcsc_col_offset_tiled*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_col_tile_indices), spmm.size_mcsc_col_offset_tiled/4*sizeof(int)));
            stat.fn_mem_use[stat.curr_config][spmm.sparsity] = (
                spmm.w1_cnt_nnz*sizeof(int) + 
                spmm.size_mcsc_col_offset_tiled*sizeof(int)
            );

            /* copy to device */
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_merged_row_indice_tiled, (void*)spmm.w1_merged_row_indice_tiled, spmm.w1_cnt_nnz*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_merged_col_offset_tiled, (void*)spmm.w1_merged_col_offset_tiled, spmm.size_mcsc_col_offset_tiled*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_col_tile_indices, (void*)spmm.col_tile_indices_, spmm.size_mcsc_col_offset_tiled/4*sizeof(int), cudaMemcpyHostToDevice));
        }

        if (ktype == TER_M_CSC || ktype == TER_TM_CSC_T) {
            //INFO("Benchmark ternary CSC Merged");
            /* allocate device memory for csc */
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_swizzled_col_indices), spmm.columns*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_merged_row_indice), spmm.w1_cnt_nnz*sizeof(int)));
            CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_merged_col_offset), spmm.size_csc_col_offset_merged*sizeof(int)));

            /* copy to device */
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_swizzled_col_indices, (void*)spmm.col_indices_, spmm.columns*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_merged_row_indice, (void*)spmm.w1_merged_row_indice, spmm.w1_cnt_nnz*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_merged_col_offset, (void*)spmm.w1_merged_col_offset, spmm.size_csc_col_offset_merged*sizeof(int), cudaMemcpyHostToDevice));
        }
        
        /* transfer w1 when necessary */
        if (ktype == BASELINE) {
            CUDA_CALL_CHECK(cudaMalloc((void**)(&spmm.dev_w1), spmm.size_w1*sizeof(int8_t)));
            CUDA_CALL_CHECK(cudaMemcpy((void*)spmm.dev_w1, (void*)spmm.host_w1.data(), spmm.size_w1*sizeof(int8_t), cudaMemcpyHostToDevice));
        }

        /* allocate device memory for matrix */
        CUDA_CALL_CHECK(cudaMalloc((void**)(&spmm.dev_x), spmm.size_x*sizeof(float)));
        CUDA_CALL_CHECK(cudaMalloc((void**)(&spmm.dev_res), spmm.size_res*sizeof(float)));
        CUDA_CALL_CHECK(cudaMemcpy((void*)spmm.dev_x, (void*)spmm.host_x.data(), spmm.size_x*sizeof(float), cudaMemcpyHostToDevice));

        stat.fn_mem_use[stat.curr_config][spmm.sparsity] += (
            spmm.size_x*sizeof(float) + 
            spmm.size_res*sizeof(float)
        );
        
        auto prekn_e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> prekn_span = prekn_e - prekn_s; // data prepare duration

        /* call cuda kernel */
        auto kernel_config = config_ter_spmm_kernel(spmm.rows, spmm.columns, spmm.inners, TILE_M, TILE_N, TILE_K, ktype);
        cudaEvent_t start, stop;
        float kn_span = 0;  // ms
        size_t free = 0;
        size_t total = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        switch (ktype)
        {
        case BASELINE:
            ter_spmm_float_baseline<<<kernel_config.first, kernel_config.second>>>(spmm.dev_x, spmm.dev_w1, spmm.dev_res, spmm.rows, spmm.columns, spmm.inners);
            break;
        case TER_CSC_BASE:
            ter_csc_spmm_float_baseline<<<kernel_config.first, kernel_config.second>>>(spmm.dev_x, 
                                                                                        dev_w1_neg_row_indices, dev_w1_neg_col_offset, 
                                                                                        dev_w1_pos_row_indices, dev_w1_pos_col_offset,
                                                                                        spmm.dev_res, 
                                                                                        spmm.rows, spmm.columns, spmm.inners);
            break;
        case TER_CSC_OPT:
            ter_csc_spmm_float_opt<TILE_M, TILE_N, TILE_K, TILE_K/TILE_N, F><<<kernel_config.first, kernel_config.second>>>(spmm.dev_x, 
                                                                                    dev_w1_neg_row_indices, dev_w1_neg_col_offset_tiled, 
                                                                                    dev_w1_pos_row_indices, dev_w1_pos_col_offset_tiled,
                                                                                    spmm.dev_res, 
                                                                                    spmm.rows, spmm.columns, spmm.inners);
            break;
        case TER_TM_CSC:
            ter_tiled_mcsc_spmm<TILE_M, TILE_N, TILE_K, 64, float4, float, int4><<<kernel_config.first, kernel_config.second>>>(
                                                                                    spmm.dev_x,
                                                                                    dev_w1_merged_row_indice_tiled, 
                                                                                    dev_w1_merged_col_offset_tiled, 
                                                                                    dev_w1_col_tile_indices,
                                                                                    spmm.dev_res,
                                                                                    spmm.rows, spmm.columns, spmm.inners);
            break;
        case TER_M_CSC:
            ter_mcsc_spmm_float_baseline<<<kernel_config.first, kernel_config.second>>>(spmm.dev_x, 
                                                                                    dev_w1_merged_row_indice, dev_w1_merged_col_offset, 
                                                                                    spmm.dev_res,
                                                                                    spmm.rows, spmm.columns, spmm.inners);
            break;
        case TER_TM_CSC_T:
            ter_tiled_mcsc_spmm_T<TILE_M, TILE_N, TILE_K, int4, int32_t, float4, float><<<kernel_config.first, kernel_config.second>>>(
                spmm.dev_x,
                dev_w1_merged_row_indice, dev_w1_merged_col_offset, 
                dev_w1_swizzled_col_indices,
                spmm.dev_res,
                spmm.rows, spmm.columns, spmm.inners
            );
            break;
        default:
            break;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kn_span, start, stop);
        stat.kn_spans[stat.curr_config][spmm.sparsity].push_back(kn_span);

        auto postkn_s = std::chrono::high_resolution_clock::now();

        CUDA_CALL_CHECK(cudaMemcpy((void*)spmm.host_res.data(), (void*)spmm.dev_res, spmm.size_res*sizeof(float), cudaMemcpyDeviceToHost));
        
        /* free */
        CUDA_CALL_CHECK(cudaFree(spmm.dev_x));
        CUDA_CALL_CHECK(cudaFree(spmm.dev_res));

        if (ktype == TER_CSC_BASE) {
            CUDA_CALL_CHECK(cudaFree(dev_w1_neg_row_indices));
            CUDA_CALL_CHECK(cudaFree(dev_w1_neg_col_offset));
            CUDA_CALL_CHECK(cudaFree(dev_w1_pos_row_indices));
            CUDA_CALL_CHECK(cudaFree(dev_w1_pos_col_offset));
        }

        if (ktype == TER_CSC_OPT) {
            CUDA_CALL_CHECK(cudaFree(dev_w1_neg_row_indices));
            CUDA_CALL_CHECK(cudaFree(dev_w1_neg_col_offset_tiled));
            CUDA_CALL_CHECK(cudaFree(dev_w1_pos_row_indices));
            CUDA_CALL_CHECK(cudaFree(dev_w1_pos_col_offset_tiled));
        }

        if (ktype == TER_TM_CSC) {
            CUDA_CALL_CHECK(cudaFree(dev_w1_merged_row_indice_tiled));
            CUDA_CALL_CHECK(cudaFree(dev_w1_merged_col_offset_tiled));
            CUDA_CALL_CHECK(cudaFree(dev_w1_col_tile_indices));
        }

        if (ktype == TER_M_CSC || ktype == TER_TM_CSC_T) {
            CUDA_CALL_CHECK(cudaFree(dev_w1_merged_row_indice));
            CUDA_CALL_CHECK(cudaFree(dev_w1_merged_col_offset));
        }

        if (ktype == BASELINE) {
            CUDA_CALL_CHECK(cudaFree(spmm.dev_w1));
        }
        
        auto fn_e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> postkn_span = fn_e - postkn_s;
        std::chrono::duration<double, std::milli> fn_span = fn_e - fn_s;
        stat.fn_spans[stat.curr_config][spmm.sparsity].push_back(fn_span.count());

        INFO(spmm.sparsity << " Memory: " << (float)stat.fn_mem_use[stat.curr_config][spmm.sparsity] / 1024 << "KB" << " Kernel: " << kn_span << "ms" << " Pre: " << prekn_span.count() << "ms" << " Post: " << postkn_span.count() << "ms" << " Function: " << fn_span.count() << "ms");

    }

private:
    std::pair<dim3, dim3> config_ter_spmm_kernel(const int M, const int N, const int K, const int TileM, const int TileN, const int TileK, KernelType ktype) {
        std::pair<dim3, dim3> config;
        switch (ktype)
        {
        case BASELINE:
        case TER_M_CSC:
        case TER_CSC_BASE:
        {
            int blkx = 32;
            int blky = 32;
            int gridx = (int)ceil(1.*N/blkx);
            int gridy = (int)ceil(1.*M/blky);
            config.first = dim3(gridx, gridy);
            config.second = dim3(blkx, blky);
            break;
        }
        case TER_CSC_OPT:
        case TER_TM_CSC:
        {
            int blkx = TileN;                           // Tile Width in N-dim
            int blky = 1;                               // default as 1
            int gridx = (int)ceil(1.*N/blkx);           // Number of Tile in N-dim: N/Tile_Width
            int gridy = (int)ceil(1.*M/blky/TileM);     // Tile Width in M dimension, workload for one block
            config.first = dim3(gridx, gridy);
            config.second = dim3(blkx, blky);
            break;
        }
        case TER_TM_CSC_T:
        {
            int blkx = TileN;                           // Tile Width in N-dim
            int blky = TileM;                               // default as 1
            int gridx = (int)ceil(1.*N/blkx);           // Number of Tile in N-dim: N/Tile_Width
            int gridy = (int)ceil(1.*M/blky);     // Tile Width in M dimension, workload for one block
            config.first = dim3(gridx, gridy);
            config.second = dim3(blkx, blky);
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