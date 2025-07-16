#include "ter_spmv.cuh"
#include "benchmarks.cuh"
#include <chrono>
#include <string>

typedef std::vector<std::vector<int>> SpMVDimsConfig;


void benchmark_cusparse_spmv(SpMVDimsConfig& dims, std::vector<float>& sparsities, int repeat) {
    Ternary_SpMV terSpMM;    
    int config_num = 0;
    for (const auto& d : dims) {
        SpMMStat stat;
        stat.curr_config = config_num++;
        for (const auto& s : sparsities) {
            TerSparseDataWrap<float> spMMData(d[0], d[1], d[2], s);
            spMMData.generate_random_array(spMMData.host_x.data(), spMMData.size_x);
            spMMData.generate_random_ternary_array(spMMData.host_w1.data(), spMMData.inners, spMMData.columns, spMMData.w1_cnt_neg, spMMData.w1_cnt_pos, spMMData.sparsity);
            // void generate_random_ternary_array(int8_t* dst, int rows, int columns, int& cnt_neg, int& cnt_pos, float sparsity) 
            spMMData.compress_normal_csc();
            INFO("Benchmark cuSparse SPMV");
            for (int i = 0; i < repeat; i++)
                terSpMM.ter_spmv_cusparse_spmv(spMMData, stat);
        }
        stat.dump("data_spmv_cusparse_");
    }
}

void benchmark_tercsc_spmv(SpMVDimsConfig& dims, std::vector<float>& sparsities, int repeat, Ternary_SpMV::KernelType ktype) {
    Ternary_SpMV terSpMM;
    int config_num = 0;
    for (const auto& d : dims) {
        SpMMStat stat;
        stat.curr_config = config_num++;
        for (const auto& s : sparsities) {
            TerSparseDataWrap<float> spMMData(d[0], d[1], d[2], s);
            spMMData.generate_random_array(spMMData.host_x.data(), spMMData.size_x);
            spMMData.generate_random_ternary_array(spMMData.host_w1.data(), spMMData.inners, spMMData.columns, spMMData.w1_cnt_neg, spMMData.w1_cnt_pos, spMMData.sparsity);
            spMMData.compress_merged_ter_csc();
            INFO("Benchmark Merged Ternary CSC SPMV");
            if (ktype == Ternary_SpMV::TER_MCSC_VEC) {
                for (int i = 0; i < repeat; i++)
                    terSpMM.ter_spmv_wrapper(spMMData, stat, ktype);
            }
        }
        stat.dump("data_spmv_tcsc_" + std::to_string(ktype)+"_");
    }
}



void unit_test_spmv(Ternary_SpMV::KernelType ktype) {
    SpMVDimsConfig dims = {  
                            // {256, 128, 512}, 
                            // {256, 256, 1024}, 
                            // {256, 512, 2048}, 
                            // {256, 1024, 4096}
                            //{1, 2048, 8192},
                            //{1, 3072, 8192},
                            {1, 4096, 14336},
                            };

    std::vector<float> sparsities = { .87f };
    Ternary_SpMV terSpMM;
    SpMMStat stat;

    int config = 0;
    for (auto& dim : dims) {

        for (float s : sparsities) {
            TerSparseDataWrap<float> spMMData(dim[0], dim[1], dim[2], s);

            spMMData.generate_random_array(spMMData.host_x.data(), spMMData.size_x);
            spMMData.generate_random_ternary_array(spMMData.host_w1.data(), spMMData.inners, spMMData.columns, spMMData.w1_cnt_neg, spMMData.w1_cnt_pos, spMMData.sparsity);
            
            INFO("SPMV with cuSparse");
            spMMData.compress_normal_csc();
            spMMData.compress_merged_ter_csc();
            // terSpMM.ter_spmv_cusparse(spMMData, stat);

            // spMMData.print<float, float>(1, spMMData.size_res, spMMData.host_res.data());

            /* cp referenced results */
            // float* res_cusparse = new float[spMMData.size_res];
            // memcpy(res_cusparse, spMMData.host_res.data(), spMMData.size_res*sizeof(float));
            // memset(spMMData.host_res.data(), 0, spMMData.size_res*sizeof(float));



            if (ktype == Ternary_SpMV::TER_MCSC_VEC) {
                INFO("SPMV with MCSC");
                terSpMM.ter_spmv_wrapper(spMMData, stat, ktype);
            }
            else if (ktype == Ternary_SpMV::BASELINE) {
                INFO("SPMV with BASELINE");
                terSpMM.ter_spmv_wrapper(spMMData, stat, ktype);
            }


            /* cp test results */
            // float* res_terspmv = new float[spMMData.size_res];
            // memcpy(res_terspmv, spMMData.host_res.data(), spMMData.size_res*sizeof(float));

            // spMMData.print<float, float>(1, spMMData.size_x, spMMData.host_x.data());
            // spMMData.print<float, float>(1, spMMData.size_res, spMMData.host_res.data());

            /* compare results */
            // int cmp = memcmp(res_terspmv, res_cusparse, spMMData.size_res*sizeof(float));
            // if (cmp) {
            //     ERROR("config" << config << ":" << s << " Results diff");
            // }
            // else
            //     INFO(s << " Results match");

            // delete res_cusparse;
        }
        config++;
    }

}

void benchmarks_spmv(int repeat = 15) {
    SpMVDimsConfig dims = {  
                            //{1, 128, 512}, 
                            //{1, 256, 1024}, 
                            //{1, 512, 2048}, 
                            {1, 1024, 4096}, 
                            {1, 2048, 8192},
                            {1, 4096, 16384},
                           //{ 1, 2048, 8192 },
                           // {1, 3072, 8192},
                           // {1, 4096, 14336},
                            };
    std::vector<float> sparsities = { .5f, .525f, .55f, .575f, .6f, .625f, .65f, .675f, .7f, .725f, .75f, .775f, .8f, .825f, .85f, .875f, .9f, .925f, .95f, .975f, };
    // {.70f, .71f, .72f, .73f, .74f, .75f, .76f, .77f, .78f, .79f,  0.80f, 0.81f, 0.82f, 0.83f, 0.84f, 0.85f, 0.86f, 0.87f, 0.88f, 0.89f, .90f, .91f, .92f, .93f, .94f, .95f, .96f, .97f, .98f, .99f, .999f, .9999f};
    
    // INFO("Benchmark SPMV with cusparse");
    benchmark_cusparse_spmv(dims, sparsities, repeat);

    INFO("Benchmark SPMV with ter csc opt");
    benchmark_tercsc_spmv(dims, sparsities, repeat, Ternary_SpMV::TER_MCSC_VEC);
}


//int main() {
//    // unit_test(Ternary_SpMV::TER_MCSC_VEC);
//    benchmarks();
//}