#include "ter_spmm.cuh"
#include "benchmarks.cuh"
#include <chrono>
#include <string>

typedef std::vector<std::vector<int>> SpMMDimsConfig;



// #define MCSC_T

void tcsc_opt_kernel_wrap(TerSparseDataWrap<float>& spMMData, SpMMStat& stat, Ternary_SpMM& terSpMM) {
#ifndef MCSC_T 
    if (spMMData.inners <= 32) {
        spMMData.compress_tiled_csc<TILE_WIDTH_K8>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N2, TILE_WIDTH_K8, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
        // terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N2, TILE_WIDTH_K8, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
    } else if (spMMData.inners <= 512) {
        spMMData.compress_tiled_csc<TILE_WIDTH_K64>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N16, TILE_WIDTH_K64, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
        // terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N16, TILE_WIDTH_K64, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
    } else if (spMMData.inners <= 1024) {
        spMMData.compress_tiled_csc<TILE_WIDTH_K128>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N32, TILE_WIDTH_K128, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
        // terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N32, TILE_WIDTH_K128, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
    } else if (spMMData.inners <= 2048) {
        spMMData.compress_tiled_csc<TILE_WIDTH_K256>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M2, TILE_WIDTH_N128, TILE_WIDTH_K256, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
        // terSpMM.ter_spmm_wrapper<TILE_WIDTH_M2, TILE_WIDTH_N64, TILE_WIDTH_K256, float4>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);

    } else if (spMMData.inners <= 4096) {
        /* config 3: TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_K256 */
        spMMData.compress_tiled_csc<TILE_WIDTH_N512>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_N512, float2>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
        // terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_K512, float2>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
    } else {
        /* config 4: TILE_WIDTH_M4, TILE_WIDTH_N1024, TILE_WIDTH_K512 */
        spMMData.compress_tiled_csc<TILE_WIDTH_K1024>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N1024, TILE_WIDTH_K1024, float>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
        // terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_K1024, TILE_WIDTH_K1024, float>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
    }
#endif
}





void mcsc_opt_kernel_wrap(TerSparseDataWrap<float>& spMMData, SpMMStat& stat, Ternary_SpMM& terSpMM) {
#ifndef MCSC_T 
    if (spMMData.inners <= 32) {
        spMMData.compress_tiled_mcsc<TILE_WIDTH_K8>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N2, TILE_WIDTH_K8, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC);
    } else if (spMMData.inners <= 512) {
        spMMData.compress_tiled_mcsc<TILE_WIDTH_K64>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N16, TILE_WIDTH_K64, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC);
    } else if (spMMData.inners <= 1024) {
        spMMData.compress_tiled_mcsc<TILE_WIDTH_K128>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N32, TILE_WIDTH_K128, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC);
    } else if (spMMData.inners <= 2048) {
        /* config 3: TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_K256 */
        spMMData.compress_tiled_mcsc<TILE_WIDTH_K512>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M16, TILE_WIDTH_N512, TILE_WIDTH_K512, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC);
    } else if (spMMData.inners <= 4096) {
        /* config 4: TILE_WIDTH_M8, TILE_WIDTH_N512, TILE_WIDTH_K512 */
        spMMData.compress_tiled_mcsc<TILE_WIDTH_K512>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M8, TILE_WIDTH_N512, TILE_WIDTH_K512, float>(spMMData, stat, Ternary_SpMM::TER_TM_CSC);
        // terSpMM.ter_spmm_wrapper<TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_K512, float2>(spMMData, stat, Ternary_SpMM::TER_CSC_OPT);
    } else if (spMMData.inners <= 8192) { 
        /* TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_K512 */
        spMMData.compress_tiled_mcsc<TILE_WIDTH_K512>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M8, TILE_WIDTH_N512, TILE_WIDTH_K512, float>(spMMData, stat, Ternary_SpMM::TER_TM_CSC);
    } else {
        spMMData.compress_tiled_mcsc<TILE_WIDTH_K1024>();
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M8, TILE_WIDTH_K1024, TILE_WIDTH_K1024, float>(spMMData, stat, Ternary_SpMM::TER_TM_CSC);
    }
#endif
}


void mcscT_opt_kernel_wrap(TerSparseDataWrap<float>& spMMData, SpMMStat& stat, Ternary_SpMM& terSpMM) {
#ifdef MCSC_T
    if (spMMData.inners <= 32) {
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M32, 1, TILE_WIDTH_K32, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC_T);
    } else if (spMMData.inners <= 512) {
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M32, 1, TILE_WIDTH_K32, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC_T);
    } else if (spMMData.inners <= 1024) {
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M32, 1, TILE_WIDTH_K32, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC_T);
    } else if (spMMData.inners <= 2048) {
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M32, 1, TILE_WIDTH_K32, float4>(spMMData, stat, Ternary_SpMM::TER_TM_CSC_T);
    } else if (spMMData.inners <= 4096) {
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M32, 1, TILE_WIDTH_K32, float>(spMMData, stat, Ternary_SpMM::TER_TM_CSC_T);
    } else if (spMMData.inners <= 8192) { 
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M16, 2, 64, float>(spMMData, stat, Ternary_SpMM::TER_TM_CSC_T);
    } else {
        terSpMM.ter_spmm_wrapper<TILE_WIDTH_M32, 1, TILE_WIDTH_K32, float>(spMMData, stat, Ternary_SpMM::TER_TM_CSC_T);
    }
#endif
}


void benchmark_cusparse_spmm(SpMMDimsConfig& dims, std::vector<float>& sparsities, int repeat) {
    Ternary_SpMM terSpMM;    
    int config_num = 0;
    for (const auto& d : dims) {
        SpMMStat stat;
        stat.curr_config = config_num++;
        for (const auto& s : sparsities) {
            TerSparseDataWrap<float> spMMData(d[0], d[1], d[2], s);
            spMMData.generate_random_array(spMMData.host_x.data(), spMMData.size_x);
            spMMData.generate_random_ternary_array(spMMData.host_w1.data(), spMMData.inners, spMMData.columns, spMMData.w1_cnt_neg, spMMData.w1_cnt_pos, spMMData.sparsity);
            spMMData.compress_normal_csc();

            for (int i = 0; i < repeat; i++)
                terSpMM.ter_spmm_cusparse_spmm(spMMData, stat);
        }
        stat.dump("data_cusparse_");
    }
}


void benchmark_tercsc_spmm(SpMMDimsConfig& dims, std::vector<float>& sparsities, int repeat, Ternary_SpMM::KernelType ktype) {
    Ternary_SpMM terSpMM;
    int config_num = 0;
    for (const auto& d : dims) {
        SpMMStat stat;
        stat.curr_config = config_num++;
        for (const auto& s : sparsities) {
            TerSparseDataWrap<float> spMMData(d[0], d[1], d[2], s);
            spMMData.generate_random_array(spMMData.host_x.data(), spMMData.size_x);
            spMMData.generate_random_ternary_array(spMMData.host_w1.data(), spMMData.inners, spMMData.columns, spMMData.w1_cnt_neg, spMMData.w1_cnt_pos, spMMData.sparsity);
            spMMData.compress_merged_ter_csc();

            if (ktype == Ternary_SpMM::TER_CSC_OPT) {
                for (int i = 0; i < repeat; i++)
                    tcsc_opt_kernel_wrap(spMMData, stat, terSpMM);
            } else if (ktype == Ternary_SpMM::TER_TM_CSC) {
                for (int i = 0; i < repeat; i++)
                    mcsc_opt_kernel_wrap(spMMData, stat, terSpMM);
            } else if (ktype == Ternary_SpMM::TER_TM_CSC_T) {
                for (int i = 0; i < repeat; i++)
                    mcscT_opt_kernel_wrap(spMMData, stat, terSpMM);
            } else {
                spMMData.compress_merged_ter_csc();
                for (int i = 0; i < repeat; i++)
                    terSpMM.ter_spmm_wrapper<1, TILE_WIDTH_N2, TILE_WIDTH_K8, float4>(spMMData, stat, ktype);
            }
            CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */
        }
        stat.dump("data_tcsc_" + std::to_string(ktype)+"_");
        CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */
    }
}

void benchmark_cublas_spmm(SpMMDimsConfig& dims, std::vector<float>& sparsities, int repeat) {
    Ternary_SpMM terSpMM;
    int config_num = 0;
    for (const auto& d : dims) {
        SpMMStat stat;
        stat.curr_config = config_num++;
        for (const auto& s : sparsities) {
            TerSparseDataWrap<float> spMMData(d[0], d[1], d[2], s);
            spMMData.generate_random_array(spMMData.host_x.data(), spMMData.size_x);
            spMMData.generate_random_uniform_ternary_array(spMMData.host_w1.data(), spMMData.inners, spMMData.columns, spMMData.w1_cnt_neg, spMMData.w1_cnt_pos, spMMData.sparsity);
            spMMData.duplicate_w();
            INFO("Benchmark cuBLAS GEMM");
            for (int i = 0; i < repeat; i++)
                terSpMM.ter_spmm_cublas_spmm(spMMData, stat);
        }
        stat.dump("data_cublas_");
    }
}


void benchmarks_spmm(int repeat = 15) {
    SpMMDimsConfig dims = {  
                            // {256, 128, 512}, 
                            // {256, 256, 1024}, 
                            // {256, 512, 2048}, 
                            {256, 1024,  4096}, 
                            {256, 2048,  8192},
                            {256, 4096, 16384},
                             //{256, 2048,  8192},
                             //{256, 3072,  8192},
                             //{256, 4096, 14336}, 
                            };

    // std::vector<float> sparsities = {.70f, .71f, .72f, .73f, .74f, .75f, .76f, .77f, .78f, .79f,  
    //                                 0.80f, 0.81f, 0.82f, 0.83f, 0.84f, 0.85f, 0.86f, 0.87f, 0.88f, 0.89f, 
    //                                 .90f, .91f, .92f, .93f, .94f, .95f, .96f, .97f, .98f, .99f, .999f};
    std::vector<float> sparsities = { .5f, .525f, .55f, .575f, .6f, .625f, .65f, .675f, .7f, .725f, .75f, .775f, .8f, .825f, .85f, .875f, .9f, .925f, .95f, .975f, };


    INFO("Benchmark SPMM with cuSparse");
    benchmark_cusparse_spmm(dims, sparsities, repeat);

    CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */

    INFO("Benchmark SPMM with cublas");
    benchmark_cublas_spmm(dims, sparsities, repeat);

    CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */


    INFO("Benchmark SPMM with ter csc baseline");
    benchmark_tercsc_spmm(dims, sparsities, repeat, Ternary_SpMM::TER_M_CSC);

    CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */

    INFO("Benchmark SPMM with ter csc opt");
#ifdef MCSC_T
    benchmark_tercsc(dims, sparsities, repeat, Ternary_SpMM::TER_TM_CSC_T);
#else
    benchmark_tercsc_spmm(dims, sparsities, repeat, Ternary_SpMM::TER_TM_CSC);
#endif


}

void unit_test_spmm(Ternary_SpMM::KernelType ktype) {
    SpMMDimsConfig dims = {  
                            // {256, 128, 512}, 
                            // {256, 256, 1024}, 
                            // {256, 512, 2048}, 
                            // {256, 1024, 4096}
                            {256, 2048, 8192}
                            // {16, 8, 32}
                            };

    std::vector<float> sparsities = {0.85f};
    Ternary_SpMM terSpMM;
    SpMMStat stat;



    int config = 0;
    for (auto& dim : dims) {

        for (float s : sparsities) {
            TerSparseDataWrap<float> spMMData(dim[0], dim[1], dim[2], s);

            spMMData.generate_random_array(spMMData.host_x.data(), spMMData.size_x);
            spMMData.generate_random_uniform_ternary_array(spMMData.host_w1.data(), spMMData.inners, spMMData.columns, spMMData.w1_cnt_neg, spMMData.w1_cnt_pos, spMMData.sparsity);
            
            INFO("SPMM with cuSparse");
            spMMData.compress_normal_csc();
            terSpMM.ter_spmm_cusparse_spmm(spMMData, stat);

            /* cp referenced results */
            float* res_cusparse = new float[spMMData.size_res];
            memcpy(res_cusparse, spMMData.host_res.data(), spMMData.size_res*sizeof(float));
            memset(spMMData.host_res.data(), 0, spMMData.size_res*sizeof(float));

            if (ktype == Ternary_SpMM::TER_TM_CSC) {
                INFO("SPMM with MCSC");
                mcsc_opt_kernel_wrap(spMMData, stat, terSpMM);
            } else if (ktype == Ternary_SpMM::TER_CSC_OPT) {
                INFO("SPMM with TCSC");
                tcsc_opt_kernel_wrap(spMMData, stat, terSpMM);
            } else if (ktype == Ternary_SpMM::TER_TM_CSC_T) {
                INFO("SPMM with TMCSC T");
                spMMData.compress_merged_ter_csc();
                terSpMM.ter_spmm_wrapper<8, 1, 16, float4>(spMMData, stat, ktype);
            } else {
                INFO("SPMM with CSC Baseline");
                spMMData.compress_merged_ter_csc();
                terSpMM.ter_spmm_wrapper<1, TILE_WIDTH_N2, TILE_WIDTH_K8, float4>(spMMData, stat, ktype);
            }

            /* cp test results */
            float* res_terspmm = new float[spMMData.size_res];
            memcpy(res_terspmm, spMMData.host_res.data(), spMMData.size_res*sizeof(float));

            // spMMData.print<float, float>(dim[0], dim[2], spMMData.host_x.data());
            // spMMData.print<int, int>(dim[1], dim[2], spMMData.w1_merged_row_indice);
            // INFO(spMMData.w1_cnt_neg << ":" << spMMData.w1_cnt_pos);
            spMMData.print<float, float>(1, 20, res_cusparse);
            spMMData.print<float, float>(1, 20, res_terspmm);


            /* compare results */
            int cmp = memcmp(res_terspmm, res_cusparse, spMMData.size_res*sizeof(float));
            if (cmp) {
                ERROR("config" << config << ":" << s << " Results diff");
            }
            else
                INFO(s << " Results match");
            
            delete res_cusparse;
            delete res_terspmm;
        }
        config++;
    }

}



//int main() {
//    // unit_test(Ternary_SpMM::TER_TM_CSC);
//    benchmarks(40);
//
//}