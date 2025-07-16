#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstdio>

#include "TCSC.hpp"
#include "initData.hpp"
#include "GEMM_CPU_INT8.hpp"
#include "GEMM_CPU_FP32.hpp"
#include "LlamaModel.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <torch/torch.h>
using namespace std;

std::vector<int64_t>  record_time(vector<string> names, vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints, std::vector<int64_t> records, FILE* fptr) {
    for (int j = 0; j < names.size(); j++) {
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timePoints[j + 1] - timePoints[j]).count();
        records[j] += ns;
        fprintf(fptr, "%lld \t", ns);
    }
    fprintf(fptr, "\n");
    return records;
}

void record_config(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, FILE* fptr) {
    fprintf(fptr, "Benchmarking M=%d K=%d N=%d Sparsity=%f Variation=%f \n", M_ROW, K_LEN, N_COL, Sparsity, Variation);
    for (int n = 0; n < names.size(); n++) {
        fprintf(fptr, names[n].c_str());
    }
    fprintf(fptr, "\n");
}

std::vector<float> print_ms_speedup(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, std::vector<float> speedups, std::vector<int64_t> records, FILE* fptr) {
    std::cout << "M=" << M_ROW << ", K=" << K_LEN << ", N=" << N_COL << ", Sparsity=" << std::fixed << std::setprecision(2) << Sparsity << " +/- " << Variation << std::endl;
    int64_t baseline = records[0];
    for (int n = 0; n < names.size(); n++) {
        float speedup = (double)baseline / (double)records[n];
        speedups.push_back(speedup);
        std::cout << names[n] << records[n] / 1000000 << "\t ms, speedup = " << std::fixed << std::setprecision(2) << speedup << std::endl;
        fprintf(fptr, "%lld\t", records[n] / 1000000);
    }
    std::cout << "\n" << std::endl;
    fprintf(fptr, "ms\n");
    for (int n = 0; n < names.size(); n++) {
        float speedup = (double)baseline / (double)records[n];
        fprintf(fptr, "%.2f\t", speedup);
    }
    fprintf(fptr, "speedup\n");
    return speedups;
}

void print_speedup_summary(vector<string> names, std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV, std::vector<float> speedups) {
    std::cout << "speedup" << std::endl;
    for (int n = 0; n < names.size(); n++) {
        std::cout << names[n];
    }
    std::cout << std::endl;
    for (int i = 0; i < Config_MKNSV.size(); i++) {
        std::cout << "M=" << std::get<0>(Config_MKNSV[i]) << " K=" << std::get<1>(Config_MKNSV[i]) << " N=" << std::get<2>(Config_MKNSV[i]) << " S=" << std::fixed << std::setprecision(2) << std::get<3>(Config_MKNSV[i]) << " +/- " << std::get<4>(Config_MKNSV[i]) << "\t";
        for (int n = 0; n < names.size(); n++)
            std::cout << speedups[i * names.size() + n] << "\t";
        std::cout << std::endl;
    }
}

std::string time_string() { 
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now(); 
    std::time_t now_c = std::chrono::system_clock::to_time_t(now); 
    std::tm* p_tm = std::localtime(&now_c); 
    if (p_tm == nullptr) {
        return "2025-01-01_00-00-00";
    }
    std::ostringstream oss;
    oss << std::put_time(p_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

int benchmark_GEMMs(){
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        // {64, 128, 32, 0.9, 0.01},
        // {256, 1024, 4096, 0.95,0.01},
        /* {256, 1024,  4096, 0.50,0.05}, // Different Sparsity
        {256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.70,0.05},
        {256, 2048,  8192, 0.90,0.05},
        {256, 4096, 16384, 0.50,0.05},
        {256, 4096, 16384, 0.70,0.05},
        {256, 4096, 16384, 0.90,0.05},
        { 32, 2048,  8192, 0.50,0.05},
        { 64, 2048,  8192, 0.50,0.05},
        {128, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.50,0.05},
       {1024, 2048,  8192, 0.50,0.05},*/
        {  1, 1024,  4096, 0.50,0.05},  
        {  1, 1024,  4096, 0.60,0.05},
        {  1, 1024,  4096, 0.70,0.05},
        {  1, 1024,  4096, 0.80,0.05},
        {  1, 1024,  4096, 0.90,0.05},
        {  1, 2048,  8192, 0.50,0.05},
        {  1, 2048,  8192, 0.60,0.05},
        {  1, 2048,  8192, 0.70,0.05},
        {  1, 2048,  8192, 0.80,0.05},
        {  1, 2048,  8192, 0.90,0.05},
        {  1, 4096, 16384, 0.50,0.05},
        {  1, 4096, 16384, 0.60,0.05},
        {  1, 4096, 16384, 0.70,0.05},
        {  1, 4096, 16384, 0.80,0.05},
        {  1, 4096, 16384, 0.90,0.05},
    };

    /*
     FILE* fptr = fopen("SpGEMM_CPU_FP32_General_Optimization_2025-04-12_19-10.txt", "w"); //Write results to
     vector<string> names = { "EigenDense\t", "GEMMOpenMP\t", "NaiveTCSC\t", "Unroll_4x4\t", "AVX2_8x4\t", "OpenMP16T\t", "MergedTCSC\t", "MTCSC_MinG4\t", "MTCSC_MidG4\t", "M_AlignG4\t", "UniformG4\t" };

     FILE* fptr = fopen("SpGEMM_CPU_FP32_colMajor_GroupMin_Sizes_2025-04-12_11-30.txt", "w");
     vector<string> names = {"EigenDense\t", "NaiveTCSC_8x4MP\t", "Merged_16x1\t", "Merged_16x1_if\t", "Mergedji_8x4\t", "Mergedji_i\t", "Mergedij\t", "Mergedij_j\t",
      "GroupMin_8xG4\t", "GroupMin_8xG8\t", "GroupMin_8xG16\t", "GroupMin_8xG32\t", 
      "GroupMin_16xG4\t", "GroupMin_16xG8\t", "GroupMin_16xG16\t", "GroupMin_16xG32\t",
      "GroupMin_32xG4\t", "GroupMin_32xG8\t", "GroupMin_32xG16\t", "GroupMin_32xG32\t" };

     FILE* fptr = fopen("SpGEMM_CPU_FP32_colMajor_Uniform_Sizes_2025-04-12_11-50.txt", "w"); 
     vector<string> names = { "EigenDense\t", "NaiveTCSC\t", "MTCSC_Min8G4\t", "MTCSC_Min32G8\t", "Uniform8G4\t", "Uniform8G8\t", "Uniform8G16\t", "Uniform8G32\t",
      "Uniform16G4\t", "Uniform16G8\t", "Uniform16G16\t", "Uniform16G32\t",  "Uniform32G4\t", "Uniform32G8\t", "Uniform32G16\t", "Uniform32G32\t"};

     FILE* fptr = fopen("SpGEMM_CPU_FP32_rowMajor_General_Optimization_2025-04-13_08-40.txt", "w");
     vector<string> names = { "EigenDense\t", "NaiveTCSC\t", "MTCSC_Min8G4\t", "MTCSC_Min32G8\t", "colGEMMOpenMP\t", "rowGEMMOpenMP\t", "rowMajorNaive\t", "rowMajorUnroll\t", 
        "rowMin1G8Cij\t", "rowMin1G8Cijj\t", "rowMin1G8Cji\t", "rowMin1G8Cjii\t" };

     FILE* fptr = fopen("SpGEMM_CPU_FP32_rowMajor_Group_Sizes_2025-04-13_11-40.txt", "w");
     vector<string> names = { "rowMin1G8\t", "rowMin2G8\t", "rowMin4G8\t", "rowMin8G8\t", "rowMin16G8\t", "rowMin1G16\t", "rowMin2G16\t", "rowMin4G16\t", "rowMin8G16\t", "rowMin16G16\t",
                           "rowMin1G32\t", "rowMin2G32\t", "rowMin4G32\t", "rowMin8G32\t", "rowMin16G32\t", "rowMin1G64\t", "rowMin2G64\t", "rowMin4G64\t", "rowMin8G64\t", "rowMin16G64\t" };
    
     FILE* fptr = fopen("SpGEMM_CPU_FP32_colMajor_AVX-512_Group_Sizes_2025-04-15_13-40.txt", "w");
      vector<string> names = { "GroupMin_16xG1\t", "GroupMin_16xG4\t", "GroupMin_16xG8\t", "GroupMin_16xG16\t", "GroupMin_16xG32\t", 
                               "GroupMin_32xG1\t", "GroupMin_32xG4\t", "GroupMin_32xG8\t", "GroupMin_32xG16\t", "GroupMin_32xG32\t",
       "Uniform16G1\t", "Uniform16G4\t", "Uniform16G8\t", "Uniform16G16\t", "Uniform16G32\t",  "Uniform32G1\t", "Uniform32G4\t", "Uniform32G8\t", "Uniform32G16\t", "Uniform32G32\t"};

    FILE* fptr = fopen("SpGEMV_CPU_FP32_rowMajor_Group_Sizes_2025-04-17_15-30.txt", "w");
    vector<string> names = { "EigenDense\t", "GroupMin_G8_AVX2\t",  "GroupMin_G16_AVX2\t",  "GroupMin_G32_AVX2\t",  "GroupMin_G64_AVX2\t", "Uniform_G8_AVX2_\t", "Uniform_G16_AVX2\t", "Uniform_G32_AVX2\t", "Uniform_G64_AVX2\t",
                     "GroupMin_G16_AVX512\t",  "GroupMin_G32_AVX512\t",  "GroupMin_G64_AVX512\t",  "GroupMin_G128_AVX512\t", "Uniform_G16_AVX512\t", "Uniform_G32_AVX512\t", "Uniform_G64_AVX512\t", "Uniform_G128_AVX512\t"};
    
    FILE* fptr = fopen("SpGEMV_CPU_FP32_rowMajor_GroupMin_Sizes_2025-04-17_22-30.txt", "w");
    vector<string> names = { "EigenDense\t", "G8_AVX2_\t",  "G16_AVX2\t",  "G32_AVX2\t",  "G64_AVX2\t", "G16_CS8_AVX2\t", "G32_CS8_AVX2\t", "G64_CS8_AVX2\t", "G64_CS8_SIMD1\t", "G64_CS8_SIMD2\t", "G64_CS8_SIMD3\t", "G128_CS8_AVX2\t",
                     "G16_AVX512\t", "G32_AVX512\t", "G64_AVX512\t", "G128_AVX512\t", "G32_CS16_AVX512\t", "G64_CS16_AVX512\t", "G128_S16_AVX512\t", "G128_CS16_SIMD1\t", "G128_CS16_SIMD2\t", "G128_CS16_SIMD3\t" };

    FILE* fptr = fopen("SpGEMV_CPU_FP32_rowMajor_Uniform_Sizes_2025-04-21_09-30.txt", "w");
    vector<string> names = { "EigenDense\t", "G8_AVX2_\t",  "G16_AVX2\t",  "G32_AVX2\t",  "G64_AVX2\t", "G16_AVX512\t", "G32_AVX512\t", "G64_AVX512\t", "G128_AVX512\t", 
        "G16_CS8_AVX2\t", "G32_CS8_AVX2\t", "G64_CS8_AVX2\t", "G128_CS8_AVX2\t", "G32_CS16_AVX512\t", "G64_CS16_AVX512\t", "G128_S16_AVX512\t" };
    */
    
    //FILE* fptr = fopen("CPU_FP32_Matrix_2048x8192_Sparsity_45-90_PyTorch_Only_2025-04-26_14-30.txt", "w");
    //vector<string> names = { "PyTorchDense\t",  "PyTorchSpCSR\t",  "PyTorchSpCSC\t" }; // , "MergedMin_32G8_AVX2\t", "MergedMin_32G16_AVX512\t", "Uniform_32G8_AVX2\t", "Uniform_32G16_AVX512\t"}; // "EigenDense\t", "EigenSparse\t",}; 
    //FILE* fptr = fopen("CPU_FP32_Matrix_2048x8192_Sparsity_45-90_Eigen_Only_2025-04-26_14-30.txt", "w");
    //vector<string> names = { "EigenDense\t", "EigenSparse\t" }; // };  
    //FILE* fptr = fopen("CPU_FP32_Matrix_4096x16384_Sparsity_45-90_TernaryLLM_EigenSparse_2025-04-26_14-30.txt", "w");
    //vector<string> names = { "EigenSparse\t", "MergedMin_32G8_AVX2\t", "MergedMin_32G16_AVX512\t", "Uniform_32G8_AVX2\t", "Uniform_32G16_AVX512\t" };
    //FILE* fptr = fopen("CPU_FP32_Matrix_4096x16384_Sparsity_45-90_TernaryLLM_PyTorch_2025-04-26_14-30.txt", "w");
    //vector<string> names = { "PyTorchDense\t", "PyTorchSpCSC\t", "MergedMin_32G8_AVX2\t", "MergedMin_32G16_AVX512\t", "Uniform_32G8_AVX2\t", "Uniform_32G16_AVX512\t" };
    //FILE* fptr = fopen("CPU_FP32_MLP_Sparsity_50-90_TernaryLLM_PyTorch_2025-05-14_10-00.txt", "w");
    //vector<string> names = { "PyTorchDense\t", "PyTorchSpCSC\t", "NaiveSiLU\t", "NaiveSiLU_Unroll16\t", "NaiveSiLU_Unroll16_LCS\t","NaiveSiLU_Unroll_AVX2\t","NaiveSiLU_Unroll_AVX512\t","NaiveSiLU_STEP16\t", "NaiveSiLU_STEP16_Unroll\t", "NaiveSiLU_STEP16_LCS\t", };
    //FILE* fptr = fopen("CPU_FP32_MLP_Sparsity_50-70-90_TernaryLLM_EigenSparse_2025-05-15_18-00.txt", "w");
    //vector<string> names = { "EigenDense\t", "Eigen_SiLU_AVX2\t", "EigenSparse\t", "MergedMin_32G8_AVX2\t", "MergedMin_32G16_AVX512\t", "Uniform_32G8_AVX2\t", "Uniform_32G16_AVX512\t" };
    //FILE* fptr = fopen("CPU_FP32_MLP_Gen_Sparsity_50-70-90_TernaryLLM_EigenSparse_2025-05-15_18-00.txt", "w");
    //vector<string> names = { "EigenDense\t", "Eigen_SiLU_AVX2\t", "EigenSparse\t", "MergedMin_G16_AVX2\t", "MergedMin_G32_AVX512\t", "MergedMin_G64_AVX2\t", "MergedMin_G64_AVX512\t","Uniform_G16_AVX2\t", "Uniform_G32_AVX512\t","Uniform_G64_AVX2\t", "Uniform_G64_AVX512\t" };
    
    /*
    Run with all the performance parameters, 10 runs for each,
    larger matrices take long to execute, so less runs for those
    */
    int NUM_RUNS = 10;
    int NUM_FUNCTIONS = 26; // 25 functions max
    std::cout << "Running performance" << std::endl;
    FILE* fptr = fopen("CPU_FP32_MLP_Gen_Sparsity_50-70-90_TernaryLLM_PyTorch_2025-06-20_11-00-static.txt", "w");
    vector<string> names = { "PyTorchDense\t", "PyTorchSpCSC\t", "MergedMin_G16_AVX2\t", "MergedMin_G32_AVX512\t", "MergedMin_G64_AVX2\t", "MergedMin_G64_AVX512\t","Uniform_G16_AVX2\t", "Uniform_G32_AVX512\t","Uniform_G64_AVX2\t", "Uniform_G64_AVX512\t" };
    std::vector<float> speedups;
    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation]: Config_MKNSV) {                    
        std::vector<float> Activation = initX<float>(M_ROW * K_LEN, 512);//Activation 
        // std::vector<float> ActivationT = transposeVector(Activation.data(), M_ROW, K_LEN);
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed
        std::vector<int8_t> WeightG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); // Gate proj
        std::vector<int8_t> WeightD = sparseWeight<int8_t>(N_COL, K_LEN, Sparsity, Variation, false, false); // Down proj
        // std::vector<int8_t> WeightAligned = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, true, false); //Weights aligned, not uniformed
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0)
        std::vector<int8_t> WeightUniformG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); // Gate proj
        std::vector<int8_t> WeightUniformD = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); // Down proj
        std::vector<float> Weight_FP32(Weight.begin(), Weight.end());
        std::vector<float> Weight_FP32G(WeightG.begin(), WeightG.end());
        std::vector<float> Weight_FP32D(WeightD.begin(), WeightD.end());
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<float> Y_CalB(M_ROW * N_COL, 0);
        // Convert vectors into *
        float* X_FP32 = Activation.data();
        int8_t* W_INT8 = Weight.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        // Ternary CSC arrays
        SparseFormat naiveTCSC = SparseFormat(W_INT8, K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);       
        MergedTCSC_Group mergedTCSC_G16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", false);
        MergedTCSC_Group mergedTCSC_G32_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false);
        MergedTCSC_Group mergedTCSC_G64_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false);
        MergedTCSC_Group uniformTCSC_G16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", false);
        MergedTCSC_Group uniformTCSC_G32 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", false);
        MergedTCSC_Group uniformTCSC_G64 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false);
        //MergedTCSC_Group mergedTCSC_G8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", true);
        //MergedTCSC_Group mergedTCSC_G16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", true);
        //MergedTCSC_Group uniformTCSC_G8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", true);
        //MergedTCSC_Group uniformTCSC_G16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", true);

        SparseFormat naiveTCSCG = SparseFormat(WeightG.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_UniformG = SparseFormat(WeightUniformG.data(), K_LEN, N_COL);
        MergedTCSC_Group mergedTCSC_G16_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 16, "min", false);
        MergedTCSC_Group mergedTCSC_G32_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 32, "min", false);
        MergedTCSC_Group mergedTCSC_G64_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 64, "min", false);
        MergedTCSC_Group uniformTCSC_G16G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 16, "uniform", false);
        MergedTCSC_Group uniformTCSC_G32G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 32, "uniform", false);
        MergedTCSC_Group uniformTCSC_G64G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 64, "uniform", false);
        //MergedTCSC_Group mergedTCSC_G8_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 8, "min", true);
        //MergedTCSC_Group mergedTCSC_G16_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 16, "min", true);
        //MergedTCSC_Group uniformTCSC_G8G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 8, "uniform", true);
        //MergedTCSC_Group uniformTCSC_G16G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 16, "uniform", true); 

        SparseFormat naiveTCSCD = SparseFormat(WeightD.data(), N_COL, K_LEN);
        SparseFormat naiveTCSC_UniformD = SparseFormat(WeightUniformD.data(), N_COL, K_LEN);
        MergedTCSC_Group mergedTCSC_G16_MinD = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 16, "min", false);
        MergedTCSC_Group mergedTCSC_G32_MinD = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 32, "min", false);
        MergedTCSC_Group mergedTCSC_G64_MinD = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 64, "min", false);
        MergedTCSC_Group uniformTCSC_G16D = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 16, "uniform", false);
        MergedTCSC_Group uniformTCSC_G32D = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 32, "uniform", false);
        MergedTCSC_Group uniformTCSC_G64D = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 64, "uniform", false);
        //MergedTCSC_Group mergedTCSC_G8_MinD = MergedTCSC_Group(naiveTCSCD, N_COL, K_LEN, 8, "min", true);
        //MergedTCSC_Group mergedTCSC_G16_MinD = MergedTCSC_Group(naiveTCSCD, N_COL, K_LEN, 16, "min", true);
        //MergedTCSC_Group uniformTCSC_G8D = MergedTCSC_Group(naiveTCSC_UniformD, N_COL, K_LEN, 8, "uniform", true);
        //MergedTCSC_Group uniformTCSC_G16D = MergedTCSC_Group(naiveTCSC_UniformD, N_COL, K_LEN, 16, "uniform", true); 

        vector<int> G16C_index(mergedTCSC_G16_Min.row_index.begin(), mergedTCSC_G16_Min.row_index.end());
        vector<int> G32C_index(mergedTCSC_G32_Min.row_index.begin(), mergedTCSC_G32_Min.row_index.end());
        vector<int> G64C_index(mergedTCSC_G64_Min.row_index.begin(), mergedTCSC_G64_Min.row_index.end());
        vector<int> UG16C_index(uniformTCSC_G16.row_index.begin(), uniformTCSC_G16.row_index.end());
        vector<int> UG32C_index(uniformTCSC_G32.row_index.begin(), uniformTCSC_G32.row_index.end());
        vector<int> UG64C_index(uniformTCSC_G64.row_index.begin(), uniformTCSC_G64.row_index.end());
        vector<int> G16C_indexG(mergedTCSC_G16_MinG.row_index.begin(), mergedTCSC_G16_MinG.row_index.end());
        vector<int> G32C_indexG(mergedTCSC_G32_MinG.row_index.begin(), mergedTCSC_G32_MinG.row_index.end());
        vector<int> G64C_indexG(mergedTCSC_G64_MinG.row_index.begin(), mergedTCSC_G64_MinG.row_index.end());
        vector<int> UG16C_indexG(uniformTCSC_G16G.row_index.begin(), uniformTCSC_G16G.row_index.end());
        vector<int> UG32C_indexG(uniformTCSC_G32G.row_index.begin(), uniformTCSC_G32G.row_index.end());
        vector<int> UG64C_indexG(uniformTCSC_G64G.row_index.begin(), uniformTCSC_G64G.row_index.end());
        vector<int> G16C_indexD(mergedTCSC_G16_MinD.row_index.begin(), mergedTCSC_G16_MinD.row_index.end());
        vector<int> G32C_indexD(mergedTCSC_G32_MinD.row_index.begin(), mergedTCSC_G32_MinD.row_index.end());
        vector<int> G64C_indexD(mergedTCSC_G64_MinD.row_index.begin(), mergedTCSC_G64_MinD.row_index.end());
        vector<int> UG16C_indexD(uniformTCSC_G16D.row_index.begin(), uniformTCSC_G16D.row_index.end());
        vector<int> UG32C_indexD(uniformTCSC_G32D.row_index.begin(), uniformTCSC_G32D.row_index.end());
        vector<int> UG64C_indexD(uniformTCSC_G64D.row_index.begin(), uniformTCSC_G64D.row_index.end());

 /*     SparseFormat naiveTCSC = SparseFormat(W_INT8, K_LEN, N_COL);
        SparseFormat naiveTCSC_Aligned = SparseFormat(WeightAligned.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);
        int* w_col_start_pos = naiveTCSC.col_start_pos.data();
        int* w_col_start_neg = naiveTCSC.col_start_neg.data();
        int16_t* w_row_index_pos = naiveTCSC.row_index_pos.data();
        int16_t* w_row_index_neg = naiveTCSC.row_index_neg.data();
        MergedTCSC mergedTCSC = MergedTCSC(naiveTCSC, K_LEN, N_COL); 

        MergedTCSC_Group(SparseFormat naiveTCSC, int K, int N, int group_size, string group_method, bool interleaved) 
        MergedTCSC_Group mergedTCSC_G4_Min   = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL,  4, "min", true);
        MergedTCSC_Group mergedTCSC_G8_Min   = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL,  8, "min", true);
        MergedTCSC_Group mergedTCSC_G16_Min  = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", true);
        MergedTCSC_Group mergedTCSC_G32_Min  = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", true);
        MergedTCSC_Group mergedTCSC_G64_Min  = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", true);
        MergedTCSC_Group mergedTCSC_G8C_Min  = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL,  8, "min", false);
        MergedTCSC_Group mergedTCSC_G16C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", false);
        MergedTCSC_Group mergedTCSC_G32C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false);
        MergedTCSC_Group mergedTCSC_G64C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false);
        MergedTCSC_Group mergedTCSC_G128C_Min= MergedTCSC_Group(naiveTCSC, K_LEN, N_COL,128, "min", false);
        MergedTCSC_Group mergedTCSC_G16CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G32CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G64CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G128CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 128, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G32CS16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false, 16);
        MergedTCSC_Group mergedTCSC_G64CS16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false, 16);
        MergedTCSC_Group mergedTCSC_G128CS16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 128, "min", false, 16);
        MergedTCSC_Group mergedTCSC_G4_Mid   = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL,  4, "mid", true);
        MergedTCSC_Group mergedTCSC_G4_Max   = MergedTCSC_Group(naiveTCSC_Aligned, K_LEN, N_COL, 4, "max", true); 
        MergedTCSC_Group uniformTCSC_G1 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 1, "uniform", true);
        MergedTCSC_Group uniformTCSC_G4 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 4, "uniform", true);
        MergedTCSC_Group uniformTCSC_G8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", true);
        MergedTCSC_Group uniformTCSC_G16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", true);
        MergedTCSC_Group uniformTCSC_G32 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", true);
        MergedTCSC_Group uniformTCSC_G64 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", true);
        MergedTCSC_Group uniformTCSC_G128 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", true);
        MergedTCSC_Group uniformTCSC_G8C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", false);
        MergedTCSC_Group uniformTCSC_G16C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", false);
        MergedTCSC_Group uniformTCSC_G32C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", false);
        MergedTCSC_Group uniformTCSC_G64C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false);
        MergedTCSC_Group uniformTCSC_G128C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", false);
        MergedTCSC_Group uniformTCSC_G16CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G32CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G64CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G128CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G32CS16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", false, 16);
        MergedTCSC_Group uniformTCSC_G64CS16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false, 16);
        MergedTCSC_Group uniformTCSC_G128CS16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", false, 16);
        vector<int> G8C_index(mergedTCSC_G8C_Min.row_index.begin(), mergedTCSC_G8C_Min.row_index.end());
        vector<int> G16C_index(mergedTCSC_G16C_Min.row_index.begin(), mergedTCSC_G16C_Min.row_index.end());
        vector<int> G32C_index(mergedTCSC_G32C_Min.row_index.begin(), mergedTCSC_G32C_Min.row_index.end());
        vector<int> G64C_index(mergedTCSC_G64C_Min.row_index.begin(), mergedTCSC_G64C_Min.row_index.end()); 
        vector<int> G128C_index(mergedTCSC_G128C_Min.row_index.begin(), mergedTCSC_G128C_Min.row_index.end());
        vector<int> G16CS8_index(mergedTCSC_G16CS8_Min.row_index.begin(), mergedTCSC_G16CS8_Min.row_index.end());
        vector<int> G32CS8_index(mergedTCSC_G32CS8_Min.row_index.begin(), mergedTCSC_G32CS8_Min.row_index.end());
        vector<int> G64CS8_index(mergedTCSC_G64CS8_Min.row_index.begin(), mergedTCSC_G64CS8_Min.row_index.end());
        vector<int> G128CS8_index(mergedTCSC_G128CS8_Min.row_index.begin(), mergedTCSC_G128CS8_Min.row_index.end());
        vector<int> G32CS16_index(mergedTCSC_G32CS16_Min.row_index.begin(), mergedTCSC_G32CS16_Min.row_index.end());
        vector<int> G64CS16_index(mergedTCSC_G64CS16_Min.row_index.begin(), mergedTCSC_G64CS16_Min.row_index.end());
        vector<int> G128CS16_index(mergedTCSC_G128CS16_Min.row_index.begin(), mergedTCSC_G128CS16_Min.row_index.end());
        vector<int> UG8C_index(uniformTCSC_G8C.row_index.begin(), uniformTCSC_G8C.row_index.end());
        vector<int> UG16C_index(uniformTCSC_G16C.row_index.begin(), uniformTCSC_G16C.row_index.end());
        vector<int> UG32C_index(uniformTCSC_G32C.row_index.begin(), uniformTCSC_G32C.row_index.end());
        vector<int> UG64C_index(uniformTCSC_G64C.row_index.begin(), uniformTCSC_G64C.row_index.end());
        vector<int> UG128C_index(uniformTCSC_G128C.row_index.begin(), uniformTCSC_G128C.row_index.end());
        vector<int> UG16CS8_index(uniformTCSC_G16CS8.row_index.begin(), uniformTCSC_G16CS8.row_index.end());
        vector<int> UG32CS8_index(uniformTCSC_G32CS8.row_index.begin(), uniformTCSC_G32CS8.row_index.end());
        vector<int> UG64CS8_index(uniformTCSC_G64CS8.row_index.begin(), uniformTCSC_G64CS8.row_index.end());
        vector<int> UG128CS8_index(uniformTCSC_G128CS8.row_index.begin(), uniformTCSC_G128CS8.row_index.end()); 
        vector<int> UG32CS16_index(uniformTCSC_G32CS16.row_index.begin(), uniformTCSC_G32CS16.row_index.end());
        vector<int> UG64CS16_index(uniformTCSC_G64CS16.row_index.begin(), uniformTCSC_G64CS16.row_index.end());
        vector<int> UG128CS16_index(uniformTCSC_G128CS16.row_index.begin(), uniformTCSC_G128CS16.row_index.end()); 
        */
        // Eigen version
        //Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32, M_ROW, K_LEN);
        //Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen(Weight_FP32.data(), K_LEN, N_COL);
        //Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenG(Weight_FP32G.data(), K_LEN, N_COL);
        //Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenD(Weight_FP32D.data(), N_COL, K_LEN);
        //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen(M_ROW, N_COL);
        //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigenG(M_ROW, N_COL);
        //Eigen::SparseMatrix<float> W_eigen_sparse = W_eigen.sparseView();
        //Eigen::SparseMatrix<float> W_eigen_sparseG = W_eigenG.sparseView();
        //Eigen::SparseMatrix<float> W_eigen_sparseD = W_eigenD.sparseView();

        torch::Tensor tensorX = torch::from_blob(Activation.data(), {M_ROW, K_LEN}, torch::kFloat32);
        torch::Tensor tensorW = torch::from_blob(Weight_FP32.data(), {K_LEN, N_COL}, torch::kFloat32);
        torch::Tensor tensorWcsc = tensorW.to_sparse_csc();
        torch::Tensor tensorWcsr = tensorW.to_sparse_csr();
        torch::Tensor tensorY    = torch::matmul(tensorX, tensorW);
        torch::Tensor tensorYcsr = torch::matmul(tensorX, tensorWcsr);
        torch::Tensor tensorYcsc = torch::matmul(tensorX, tensorWcsc);

        torch::Tensor tensorWG = torch::from_blob(Weight_FP32.data(), { K_LEN, N_COL }, torch::kFloat32);
        torch::Tensor tensorWcscG = tensorWG.to_sparse_csc();
        torch::Tensor tensorWD = torch::from_blob(Weight_FP32.data(), { N_COL, K_LEN }, torch::kFloat32);
        torch::Tensor tensorWcscD = tensorWD.to_sparse_csc();
        torch::Tensor tensorG = torch::matmul(tensorX, tensorWG);
        torch::Tensor tensorGcsc = torch::matmul(tensorX, tensorWcscG);

        std::vector<int64_t> records(NUM_FUNCTIONS, 0);

        fprintf(fptr, "Benchmarking M=%d K=%d N=%d Sparsity=%f Variation=%f \n", M_ROW, K_LEN, N_COL, Sparsity, Variation);
        for (int n = 0; n < names.size(); n++) {
            fprintf(fptr, names[n].c_str());
        }
        fprintf(fptr, "\n");
        for (int i = 0; i < NUM_RUNS; i++) {
            vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
            int j = 0; 

            /* Benchmark PyTorch, Eigen, colMajor Ternary GEMM 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_64xG4_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Y_FP32, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP(X_FP32, uniformTCSC.metadata[0], uniformTCSC.row_index.data(), Y_FP32, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_64xG4_AVX2_OpenMP(Activation.data(), mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX2_OpenMP(Activation.data(), uniformTCSC.metadata[0], uniformTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP(Activation.data(), uniformTCSC.metadata[0], uniformTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;*/

            /* Benchmark PyTorch, Eigen, colMajor Ternary GEMM - Llama MLP */
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            tensorY = torch::matmul(tensorX, tensorW); 
            tensorG = torch::matmul(tensorX, tensorWG);
            // cout << tensorG.sizes() << tensorY.sizes()<<endl;
            tensorY = torch::mul(tensorY, torch::silu(tensorG));
            tensorY = torch::matmul(tensorY, tensorWD);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            tensorYcsc = torch::matmul(tensorX, tensorWcsc); tensorYcsc.sizes();
            tensorGcsc = torch::matmul(tensorX, tensorWcscG);
            tensorYcsc = torch::mul(tensorYcsc, torch::silu(tensorGcsc));
            tensorYcsc = torch::matmul(tensorYcsc, tensorWcscD);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            
            // cout << "Eigen Nb Threads =" << Eigen::nbThreads() << endl; // Verified to be 16 threads = max threads available
            //Y_eigen = X_eigen * W_eigen; 
            //Y_eigenG = X_eigen * W_eigenG;
            //Y_eigenG = Y_eigenG.array() * Y_eigen.array() / (1.0f + (-Y_eigenG).array().exp());
            //Y_eigen = Y_eigenG * W_eigenD;
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //Y_eigen = X_eigen * W_eigen;
            //Y_eigenG = X_eigen * W_eigenG;
            //Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Y_eigenG.data(), Y_eigen.data());
            //Y_eigen = Y_eigenG * W_eigenD;
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //Y_eigen = X_eigen * W_eigen_sparse;
            //Y_eigenG = X_eigen * W_eigen_sparseG;
            //Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Y_eigenG.data(), Y_eigen.data());
            //Y_eigen = Y_eigenG * W_eigen_sparseD;
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), G16C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X_FP32, mergedTCSC_G16_MinG.metadata.data(), G16C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(Ra, mergedTCSC_G16_MinD.metadata.data(), G16C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), G32C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(X_FP32, mergedTCSC_G32_MinG.metadata.data(), G32C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX512(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(Ra, mergedTCSC_G32_MinD.metadata.data(), G32C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64_MinG.metadata.data(), G64C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(Ra, mergedTCSC_G64_MinD.metadata.data(), G64C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X_FP32, mergedTCSC_G64_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X_FP32, mergedTCSC_G64_MinG.metadata.data(), G64C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX512(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(Ra, mergedTCSC_G64_MinD.metadata.data(), G64C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], UG16C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(X_FP32, uniformTCSC_G16G.metadata[0], UG16C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(Ra, uniformTCSC_G16D.metadata[0], UG16C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], UG32C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(X_FP32, uniformTCSC_G32G.metadata[0], UG32C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX512(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(Ra, uniformTCSC_G32D.metadata[0], UG32C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X_FP32, uniformTCSC_G64.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X_FP32, uniformTCSC_G64G.metadata[0], UG64C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(Ra, uniformTCSC_G64D.metadata[0], UG64C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X_FP32, uniformTCSC_G64.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X_FP32, uniformTCSC_G64G.metadata[0], UG64C_indexG.data(), Ra, N_COL, K_LEN);
            Naive_SiLU_Dot_Unroll_AVX512(M_ROW* N_COL, Ra, Rb);
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(Ra, uniformTCSC_G64D.metadata[0], UG64C_indexD.data(), Rb, K_LEN, N_COL);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            //GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_MinG.metadata.data(), mergedTCSC_G8_MinG.row_index.data(), Ra, M_ROW, N_COL, K_LEN);
            //Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Ra, Rb);
            //GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(Ra, mergedTCSC_G8_MinD.metadata.data(), mergedTCSC_G8_MinD.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            //GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_MinG.metadata.data(), mergedTCSC_G16_MinG.row_index.data(), Ra, M_ROW, N_COL, K_LEN);
            //Naive_SiLU_Dot_Unroll_AVX512(M_ROW* N_COL, Ra, Rb);
            //GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(Ra, mergedTCSC_G16_MinD.metadata.data(), mergedTCSC_G16_MinD.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8G.metadata[0], uniformTCSC_G8G.row_index.data(), Ra, M_ROW, N_COL, K_LEN);  
            //Naive_SiLU_Dot_Unroll_AVX2(M_ROW* N_COL, Ra, Rb);
            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(Ra, uniformTCSC_G8D.metadata[0], uniformTCSC_G8D.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16G.metadata[0], uniformTCSC_G16G.row_index.data(), Ra, M_ROW, N_COL, K_LEN); 
            //Naive_SiLU_Dot_Unroll_AVX512(M_ROW* N_COL, Ra, Rb);
            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(Ra, uniformTCSC_G16D.metadata[0], uniformTCSC_G16D.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            /* Benchmark PyTorch, Eigen, colMajor Ternary GEMM  
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            tensorY = torch::matmul(tensorX, tensorW); tensorY.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //tensorYcsr = torch::matmul(tensorX, tensorWcsr); tensorYcsr.sizes();
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            tensorYcsc = torch::matmul(tensorX, tensorWcsc); tensorYcsc.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //Y_eigen = X_eigen * W_eigen; 
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //Y_eigen = X_eigen * W_eigen_sparse;
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++; */

            /* General Optimizations
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            result_eigen = X_eigen * W1_eigen; 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_Direct_OpenMP(X_FP32, W_INT8, Ra, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            // GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            // timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_4x4_Unroll(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMid_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Mid.metadata.data(), mergedTCSC_G4_Mid.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Aligned_GroupMax_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Max.metadata.data(), mergedTCSC_G4_Max.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG4_AVX2_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++; */

            /* ColMajor MergedMin Exploration 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            result_eigen = X_eigen * W1_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            // GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            // timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            // GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            // timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            // GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            // timePoints[j] = std::chrono::high_resolution_clock::now(); j++;            
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            // GEMM_CPU_FP32_colMajor_TCSC_Merged_8x1_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            // timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_if_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMPi(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMPj(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
           
            /* Explore colMajor Uniform 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            result_eigen = X_eigen * W1_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG4_AVX2_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG16_AVX2_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG32_AVX2_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG4_AVX2_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG16_AVX2_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG32_AVX2_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX2_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX2_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX2_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */

            /* RowMajor General Optimizations 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            result_eigen = X_eigen * W1_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_Direct_OpenMP(X_FP32, W_INT8, Ra, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_Direct_OpenMP(X_FP32, W_INT8, Ra, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_4x4_Unroll_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPij(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPijj(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPji(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPjii(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */

            /* Explore RowMajor Sizes 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;                                        
            */

            /* AVX-512 explorations 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_16xG1_AVX512_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG4_AVX512_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG8_AVX512_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG32_AVX512_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Merged_32xG1_AVX512_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX512_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX512_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX512_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG1_AVX512_OpenMP(X_FP32, uniformTCSC_G1.metadata[0], uniformTCSC_G1.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG4_AVX512_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG8_AVX512_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG32_AVX512_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG1_AVX512_OpenMP(X_FP32, uniformTCSC_G1.metadata[0], uniformTCSC_G1.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX512_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX512_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX512_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */

            /* GEMV GroupMin and Uniform in normal contineous TCSC 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            result_eigen = X_eigen * W1_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);      
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX512_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP(X_FP32, mergedTCSC_G128C_Min.metadata.data(), G128C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP(X_FP32, uniformTCSC_G8C.metadata[0], UG8C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(X_FP32, uniformTCSC_G16C.metadata[0], UG16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX2_OpenMP(X_FP32, uniformTCSC_G32C.metadata[0], UG32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X_FP32, uniformTCSC_G64C.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX512_OpenMP(X_FP32, uniformTCSC_G16C.metadata[0], UG16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(X_FP32, uniformTCSC_G32C.metadata[0], UG32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X_FP32, uniformTCSC_G64C.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP(X_FP32, uniformTCSC_G128C.metadata[0], UG128C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */

            /* GEMV GroupMin Only 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            result_eigen = X_eigen * W1_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_CS8_AVX2_OpenMP(X_FP32, mergedTCSC_G16CS8_Min.metadata.data(), G16CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_CS8_AVX2_OpenMP(X_FP32, mergedTCSC_G32CS8_Min.metadata.data(), G32CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_AVX2_OpenMP(X_FP32, mergedTCSC_G64CS8_Min.metadata.data(), G64CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD1_AVX2_OpenMP(X_FP32, mergedTCSC_G64CS8_Min.metadata.data(), G64CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD2_AVX2_OpenMP(X_FP32, mergedTCSC_G64CS8_Min.metadata.data(), G64CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD3_AVX2_OpenMP(X_FP32, mergedTCSC_G64CS8_Min.metadata.data(), G64CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS8_AVX2_OpenMP(X_FP32, mergedTCSC_G128CS8_Min.metadata.data(), G128CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX512_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP(X_FP32, mergedTCSC_G128C_Min.metadata.data(), G128C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_CS16_AVX512_OpenMP(X_FP32, mergedTCSC_G32CS16_Min.metadata.data(), G32CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS16_AVX512_OpenMP(X_FP32, mergedTCSC_G64CS16_Min.metadata.data(), G64CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_AVX512_OpenMP(X_FP32, mergedTCSC_G128CS16_Min.metadata.data(), G128CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD1_AVX512_OpenMP(X_FP32, mergedTCSC_G128CS16_Min.metadata.data(), G128CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD2_AVX512_OpenMP(X_FP32, mergedTCSC_G128CS16_Min.metadata.data(), G128CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD3_AVX512_OpenMP(X_FP32, mergedTCSC_G128CS16_Min.metadata.data(), G128CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */

            /* GEMV - Uniform 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            result_eigen = X_eigen * W1_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP(X_FP32, uniformTCSC_G8C.metadata[0], UG8C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(X_FP32, uniformTCSC_G16C.metadata[0], UG16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX2_OpenMP(X_FP32, uniformTCSC_G32C.metadata[0], UG32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X_FP32, uniformTCSC_G64C.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX512_OpenMP(X_FP32, uniformTCSC_G16C.metadata[0], UG16C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(X_FP32, uniformTCSC_G32C.metadata[0], UG32C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X_FP32, uniformTCSC_G64C.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP(X_FP32, uniformTCSC_G128C.metadata[0], UG128C_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_CS8_AVX2_OpenMP(X_FP32, uniformTCSC_G16CS8.metadata[0], UG16CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_CS8_AVX2_OpenMP(X_FP32, uniformTCSC_G32CS8.metadata[0], UG32CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_CS8_AVX2_OpenMP(X_FP32, uniformTCSC_G64CS8.metadata[0], UG64CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_CS8_AVX2_OpenMP(X_FP32, uniformTCSC_G128CS8.metadata[0], UG128CS8_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_CS16_AVX512_OpenMP(X_FP32, uniformTCSC_G32CS16.metadata[0], UG32CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_CS16_AVX512_OpenMP(X_FP32, uniformTCSC_G64CS16.metadata[0], UG64CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_CS16_AVX512_OpenMP(X_FP32, uniformTCSC_G128CS16.metadata[0], UG128CS16_index.data(), Rb, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */


            
            record_time(names, timePoints, records, fptr);
        }
        print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}

int benchmark_FP32_GEMM_General_Optimizations(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    // *********** File Name and Function Names *************
    string file_name = "GEMM_CPU_FP32_General_Optimizations_" + time_string() + ".txt";
    vector<string> names = { "EigenDense\t", "GEMMOpenMP\t", "NaiveTCSC\t", "Unroll_4x4\t", "AVX2_8x4\t", "OpenMP16T\t", "MergedTCSC\t",                           
                             "Merged_16x1\t", "Merged_16x1_if\t", "Mergedji_8x4j\t", "Mergedji_MPi\t", "Mergedij_MPi\t", "Mergedij_MPj\t",
                             "GroupMin_G4\t", "GroupMidPad_G4\t", "GroupMaxAlignG4\t", "UniformTCSC_G4\t", };
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
    {256,  128,  512, 0.5, 0.05}, // Different K & N
  /*{256,  256, 1024, 0.5, 0.05},
    {256,  512, 2048, 0.5, 0.05},
    {256, 1024, 4096, 0.5, 0.05},
    {256, 2048, 8192, 0.5, 0.05},
    {256,  512,  128, 0.5, 0.05}, // Different N & K
    {256, 1024,  256, 0.5, 0.05},
    {256, 2048,  512, 0.5, 0.05},
    {256, 4096, 1024, 0.5, 0.05},
    {256, 8192, 2048, 0.5, 0.05},*/
    };

    std::vector<float> speedups;
    FILE* fptr = fopen(file_name.c_str(), "w");
    std::cout << "Running " << file_name << std::endl;
    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {
        // *********** Data Initialization *************
        std::vector<float> Activation = initX<float>(M_ROW * K_LEN, 512);//Activation 
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed 
        std::vector<int8_t> WeightAligned = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, true, false); //Weights aligned, not uniformed
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0) 
        std::vector<float> Weight_FP32(Weight.begin(), Weight.end());
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication

        float* X_FP32 = Activation.data();// Convert vectors into *
        int8_t* W_INT8 = Weight.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32, M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen(Weight_FP32.data(), K_LEN, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen(M_ROW, N_COL);
        Eigen::SparseMatrix<float> W_eigen_sparse = W_eigen.sparseView();

        SparseFormat naiveTCSC = SparseFormat(W_INT8, K_LEN, N_COL);
        SparseFormat naiveTCSC_Aligned = SparseFormat(WeightAligned.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);
        int* w_col_start_pos = naiveTCSC.col_start_pos.data();
        int* w_col_start_neg = naiveTCSC.col_start_neg.data();
        int16_t* w_row_index_pos = naiveTCSC.row_index_pos.data();
        int16_t* w_row_index_neg = naiveTCSC.row_index_neg.data();
        MergedTCSC mergedTCSC = MergedTCSC(naiveTCSC, K_LEN, N_COL);
        MergedTCSC_Group mergedTCSC_G4_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 4, "min", true);
        MergedTCSC_Group mergedTCSC_G4_Mid = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 4, "mid", true);
        MergedTCSC_Group mergedTCSC_G4_Max = MergedTCSC_Group(naiveTCSC_Aligned, K_LEN, N_COL, 4, "max", true);
        MergedTCSC_Group uniformTCSC_G4    = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 4, "uniform", true);       
       
        std::vector<int64_t> records(NUM_FUNCTIONS, 0);
        record_config(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, fptr);
        for (int i = 0; i < NUM_RUNS; i++) {
            vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
            int j = 0; 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // *********** Benchmark Functions *************
            Y_eigen = X_eigen * W_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_Direct_OpenMP(X_FP32, W_INT8, Ra, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_4x4_Unroll(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_if_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMPi(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMPj(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;  

            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMid_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Mid.metadata.data(), mergedTCSC_G4_Mid.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Aligned_GroupMax_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Max.metadata.data(), mergedTCSC_G4_Max.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG4_AVX2_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;


            records = record_time(names, timePoints, records, fptr);
        }
        speedups = print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}


int benchmark_FP32_Matrix_Level(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    // *********** File Name and Function Names *************
    string file_name = "GEMM_CPU_FP32_Matrix_Level_" + time_string() + ".txt";
    vector<string> names = { "PyTorchDense\t",  "PyTorchSpCSR\t",  "PyTorchSpCSC\t", "EigenDense\t", "EigenSparse\t", 
                             "MergedMin_32G8_AVX2\t", "MergedMin_32G16_AVX512\t", "Uniform_32G8_AVX2\t", "Uniform_32G16_AVX512\t"};
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        {256, 1024,  4096, 0.50,0.05}, // Different Sparsity
      /*{256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.70,0.05},
        {256, 2048,  8192, 0.90,0.05},
        {256, 4096, 16384, 0.50,0.05},
        {256, 4096, 16384, 0.70,0.05},
        {256, 4096, 16384, 0.90,0.05},*/
    };

    std::vector<float> speedups;
    FILE* fptr = fopen(file_name.c_str(), "w");
    std::cout << "Running " << file_name << std::endl;
    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {
        // *********** Data Initialization *************
        std::vector<float> Activation = initX<float>(M_ROW * K_LEN, 512);//Activation 
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed 
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0) 
        std::vector<float> Weight_FP32(Weight.begin(), Weight.end());
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication

        float* X_FP32 = Activation.data();// Convert vectors into *
        int8_t* W_INT8 = Weight.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32, M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen(Weight_FP32.data(), K_LEN, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen(M_ROW, N_COL);
        Eigen::SparseMatrix<float> W_eigen_sparse = W_eigen.sparseView();

        SparseFormat naiveTCSC = SparseFormat(W_INT8, K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);
        MergedTCSC_Group mergedTCSC_G8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", true); 
        MergedTCSC_Group mergedTCSC_G16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", true);
        MergedTCSC_Group uniformTCSC_G8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", true);     
        MergedTCSC_Group uniformTCSC_G16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", true);

        torch::Tensor tensorX = torch::from_blob(Activation.data(), { M_ROW, K_LEN }, torch::kFloat32);
        torch::Tensor tensorW = torch::from_blob(Weight_FP32.data(), { K_LEN, N_COL }, torch::kFloat32);
        torch::Tensor tensorWcsc = tensorW.to_sparse_csc();
        torch::Tensor tensorWcsr = tensorW.to_sparse_csr();
        torch::Tensor tensorY = torch::matmul(tensorX, tensorW);
        torch::Tensor tensorYcsr = torch::matmul(tensorX, tensorWcsr);
        torch::Tensor tensorYcsc = torch::matmul(tensorX, tensorWcsc);

        std::vector<int64_t> records(NUM_FUNCTIONS, 0);
        record_config(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, fptr);
        for (int i = 0; i < NUM_RUNS; i++) {
            vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
            int j = 0;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // *********** Benchmark Functions ************* 
            tensorY = torch::matmul(tensorX, tensorW); tensorY.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            tensorYcsr = torch::matmul(tensorX, tensorWcsr); tensorYcsr.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            tensorYcsc = torch::matmul(tensorX, tensorWcsc); tensorYcsc.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // Note that Eigen speed will be much lower if compiled with PyTorch... 
            Y_eigen = X_eigen * W_eigen; 
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_eigen = X_eigen * W_eigen_sparse;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++; 


            records = record_time(names, timePoints, records, fptr);
        }
        speedups = print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}

int benchmark_FP32_Layer_Level(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    // *********** File Name and Function Names *************
    string file_name = "GEMM_CPU_FP32_Layer_Level_LlamaMLP_" + time_string() + ".txt";
    vector<string> names = { "PyTorchDense\t",  "PyTorchSpCSC\t", "EigenSparse\t", "MergedMin_32G8_AVX2\t", "MergedMin_32G16_AVX512\t", "Uniform_32G8_AVX2\t", "Uniform_32G16_AVX512\t" };

    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        {256, 1024,  4096, 0.50,0.05}, // Different Sparsity
      /*{256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.70,0.05},
        {256, 2048,  8192, 0.90,0.05},
        {256, 4096, 16384, 0.50,0.05},
        {256, 4096, 16384, 0.70,0.05},
        {256, 4096, 16384, 0.90,0.05},
        {  1, 1024,  4096, 0.50,0.05}, // Different Sparsity
        {  1, 1024,  4096, 0.70,0.05},
        {  1, 1024,  4096, 0.90,0.05},
        {  1, 2048,  8192, 0.50,0.05},
        {  1, 2048,  8192, 0.70,0.05},
        {  1, 2048,  8192, 0.90,0.05},
        {  1, 4096, 16384, 0.50,0.05},
        {  1, 4096, 16384, 0.70,0.05},*/ 
        {  1, 4096, 16384, 0.90,0.05},
    };

    std::vector<float> speedups;
    FILE* fptr = fopen(file_name.c_str(), "w");
    std::cout << "Running " << file_name << std::endl;
    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {
        // *********** Data Initialization *************
        std::vector<float> Activation = initX<float>(M_ROW * K_LEN, 512);//Activation 
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed 
        std::vector<int8_t> WeightG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); // Gate proj
        std::vector<int8_t> WeightD = sparseWeight<int8_t>(N_COL, K_LEN, Sparsity, Variation, false, false); // Down proj
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0) 
        std::vector<int8_t> WeightUniformG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); // Gate proj
        std::vector<int8_t> WeightUniformD = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); // Down proj

        std::vector<float> Weight_FP32(Weight.begin(), Weight.end());
        std::vector<float> Weight_FP32G(WeightG.begin(), WeightG.end());
        std::vector<float> Weight_FP32D(WeightD.begin(), WeightD.end());
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication

        float* X_FP32 = Activation.data();// Convert vectors into *
        int8_t* W_INT8 = Weight.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32, M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen(Weight_FP32.data(), K_LEN, N_COL);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenG(Weight_FP32G.data(), K_LEN, N_COL);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenD(Weight_FP32D.data(), N_COL, K_LEN);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen(M_ROW, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigenG(M_ROW, N_COL);
        Eigen::SparseMatrix<float> W_eigen_sparse = W_eigen.sparseView();
        Eigen::SparseMatrix<float> W_eigen_sparseG = W_eigenG.sparseView();
        Eigen::SparseMatrix<float> W_eigen_sparseD = W_eigenD.sparseView();


        SparseFormat naiveTCSC = SparseFormat(W_INT8, K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);
        MergedTCSC_Group mergedTCSC_G8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", true);
        MergedTCSC_Group mergedTCSC_G16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", true);
        MergedTCSC_Group uniformTCSC_G8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", true);
        MergedTCSC_Group uniformTCSC_G16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", true);

        SparseFormat naiveTCSCG = SparseFormat(WeightG.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_UniformG = SparseFormat(WeightUniformG.data(), K_LEN, N_COL);
        MergedTCSC_Group mergedTCSC_G8_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 8, "min", true);
        MergedTCSC_Group mergedTCSC_G16_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 16, "min", true);
        MergedTCSC_Group uniformTCSC_G8G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 8, "uniform", true);
        MergedTCSC_Group uniformTCSC_G16G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 16, "uniform", true);

        SparseFormat naiveTCSCD = SparseFormat(WeightD.data(), N_COL, K_LEN);
        SparseFormat naiveTCSC_UniformD = SparseFormat(WeightUniformD.data(), N_COL, K_LEN);
        MergedTCSC_Group mergedTCSC_G8_MinD = MergedTCSC_Group(naiveTCSCD, N_COL, K_LEN, 8, "min", true);
        MergedTCSC_Group mergedTCSC_G16_MinD = MergedTCSC_Group(naiveTCSCD, N_COL, K_LEN, 16, "min", true);
        MergedTCSC_Group uniformTCSC_G8D = MergedTCSC_Group(naiveTCSC_UniformD, N_COL, K_LEN, 8, "uniform", true);
        MergedTCSC_Group uniformTCSC_G16D = MergedTCSC_Group(naiveTCSC_UniformD, N_COL, K_LEN, 16, "uniform", true);

        MergedTCSC_Group mergedTCSC_G64_Min  = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false);         
        MergedTCSC_Group mergedTCSC_G64_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 64, "min", false);       
        MergedTCSC_Group mergedTCSC_G64_MinD = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 64, "min", false); 
        MergedTCSC_Group uniformTCSC_G64  = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false);
        MergedTCSC_Group uniformTCSC_G64G = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 64, "uniform", false); 
        MergedTCSC_Group uniformTCSC_G64D = MergedTCSC_Group(naiveTCSC_UniformG, K_LEN, N_COL, 64, "uniform", false);
        vector<int> G64C_index(  mergedTCSC_G64_Min.row_index.begin(), mergedTCSC_G64_Min.row_index.end());       
        vector<int> G64C_indexG(mergedTCSC_G64_MinG.row_index.begin(), mergedTCSC_G64_MinG.row_index.end()); 
        vector<int> G64C_indexD(mergedTCSC_G64_MinD.row_index.begin(), mergedTCSC_G64_MinD.row_index.end());
        vector<int> UG64C_index(  uniformTCSC_G64.row_index.begin(), uniformTCSC_G64.row_index.end()); 
        vector<int> UG64C_indexG(uniformTCSC_G64G.row_index.begin(), uniformTCSC_G64G.row_index.end()); 
        vector<int> UG64C_indexD(uniformTCSC_G64D.row_index.begin(), uniformTCSC_G64D.row_index.end());


        torch::Tensor tensorX = torch::from_blob(Activation.data(), { M_ROW, K_LEN }, torch::kFloat32);
        torch::Tensor tensorW = torch::from_blob(Weight_FP32.data(), { K_LEN, N_COL }, torch::kFloat32);
        torch::Tensor tensorWcsc = tensorW.to_sparse_csc();
        torch::Tensor tensorWcsr = tensorW.to_sparse_csr();
        torch::Tensor tensorY = torch::matmul(tensorX, tensorW);
        torch::Tensor tensorYcsr = torch::matmul(tensorX, tensorWcsr);
        torch::Tensor tensorYcsc = torch::matmul(tensorX, tensorWcsc);

        torch::Tensor tensorWG = torch::from_blob(Weight_FP32.data(), { K_LEN, N_COL }, torch::kFloat32);
        torch::Tensor tensorWcscG = tensorWG.to_sparse_csc();
        torch::Tensor tensorWD = torch::from_blob(Weight_FP32.data(), { N_COL, K_LEN }, torch::kFloat32);
        torch::Tensor tensorWcscD = tensorWD.to_sparse_csc();

        std::vector<int64_t> records(NUM_FUNCTIONS, 0);
        record_config(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, fptr);
        for (int i = 0; i < NUM_RUNS; i++) {
            vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
            int j = 0;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // *********** Benchmark Functions ************* 
            tensorY = torch::matmul(tensorX, tensorW);
            torch::Tensor tensorG = torch::matmul(tensorX, tensorWG);
            tensorY = torch::mul(tensorY, torch::silu(tensorG));
            tensorY = torch::matmul(tensorY, tensorWD);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            tensorYcsc = torch::matmul(tensorX, tensorWcsc); tensorYcsc.sizes();
            torch::Tensor tensorGcsc = torch::matmul(tensorX, tensorWcscG);
            tensorYcsc = torch::mul(tensorYcsc, torch::silu(tensorGcsc));
            tensorYcsc = torch::matmul(tensorYcsc, tensorWcscD);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // Note that Eigen speed will be much lower if compiled with PyTorch... 
            // cout << "Eigen Nb Threads =" << Eigen::nbThreads() << endl; // Verified to be 16 threads = max threads available
            Y_eigen = X_eigen * W_eigen_sparse; 
            Y_eigenG = X_eigen * W_eigen_sparseG;
            Y_eigenG = Y_eigenG.array() * Y_eigen.array() / (1.0f + (-Y_eigenG).array().exp());
            Y_eigen = Y_eigenG * W_eigen_sparseD;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            if (M_ROW > 1) {
                GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
                GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_MinG.metadata.data(), mergedTCSC_G8_MinG.row_index.data(), Ra, M_ROW, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ra, Rb);
                GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(Ra, mergedTCSC_G8_MinD.metadata.data(), mergedTCSC_G8_MinD.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

                GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
                GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_MinG.metadata.data(), mergedTCSC_G16_MinG.row_index.data(), Ra, M_ROW, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ra, Rb);
                GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(Ra, mergedTCSC_G16_MinD.metadata.data(), mergedTCSC_G16_MinD.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

                GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
                GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8G.metadata[0], uniformTCSC_G8G.row_index.data(), Ra, M_ROW, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ra, Rb);
                GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(Ra, uniformTCSC_G8D.metadata[0], uniformTCSC_G8D.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

                GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
                GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16G.metadata[0], uniformTCSC_G16G.row_index.data(), Ra, M_ROW, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ra, Rb);
                GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(Ra, uniformTCSC_G16D.metadata[0], uniformTCSC_G16D.row_index.data(), Rb, M_ROW, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            }
            else {
                GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
                GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64_MinG.metadata.data(), G64C_indexG.data(), Ra, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ra, Rb);
                GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(Ra, mergedTCSC_G64_MinD.metadata.data(), G64C_indexD.data(), Rb, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

                GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X_FP32, mergedTCSC_G64_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
                GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X_FP32, mergedTCSC_G64_MinG.metadata.data(), G64C_indexG.data(), Ra, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ra, Rb);
                GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(Ra, mergedTCSC_G64_MinD.metadata.data(), G64C_indexD.data(), Rb, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

                GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X_FP32, uniformTCSC_G64.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
                GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X_FP32, uniformTCSC_G64G.metadata[0], UG64C_indexG.data(), Ra, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ra, Rb);
                GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(Ra, uniformTCSC_G64D.metadata[0], UG64C_indexD.data(), Rb, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

                GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X_FP32, uniformTCSC_G64.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
                GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X_FP32, uniformTCSC_G64G.metadata[0], UG64C_indexG.data(), Ra, N_COL, K_LEN);
                Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ra, Rb);
                GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(Ra, uniformTCSC_G64D.metadata[0], UG64C_indexD.data(), Rb, K_LEN, N_COL);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            }


            records = record_time(names, timePoints, records, fptr);
        }
        speedups = print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}

int benchmark_FP32_Llama_Model(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    std::vector<std::tuple<float, float>> Config_SV = {
        {0.50,0.05},
        /*{0.60,0.05},
        {0.70,0.05},
        {0.80,0.05},*/
        {0.90,0.05},
    };
    std::vector<std::tuple<bool, bool, bool>> Config_UniformAVX512Gen = {
        {false,false,  true},
        {false,false, false},
        /*{false, true,  true},
        {false, true, false},
        {true, false,  true},
        {true, false, false},
        {true,  true,  true},
        {true,  true, false},*/
    };
    std::vector<std::tuple<int, int, int, int, int>> Config_LKNHH = {
        {  16, 2048,  8192, 32, 8},
        // {  28, 3072,  8192, 24, 8},
        // {  32, 4096, 14336, 32, 8},
    };
    std::vector<std::tuple<int, int>> Config_BS = {
        //{  1,    1},
        //{  1,   64},
        //{  1,  128},
        {  1,  256},
        //{  1,  512},
        //{  1, 1024},
    };

    std::cout << "Running performance" << std::endl;
    string file_name = "GEMM_CPU_FP32_Layer_Level_LlamaMLP_" + time_string() + ".txt";
    FILE* fptr = fopen(file_name.c_str(), "w");
    vector<string> names = { "TernaryLlama\t", };
    std::vector<float> speedups;
    for (const auto& [LAYERS, K_LEN, N_COL, QHEADS, KVHEADS] : Config_LKNHH) {
        for (const auto& [BS, M_ROW] : Config_BS) {
            TernaryLlamaModel<int16_t> model = TernaryLlamaModel<int16_t>(LAYERS, QHEADS, KVHEADS, K_LEN, N_COL, 0.5, 0.05, false, true, false);
            torch::Tensor X = torch::rand({ BS, M_ROW, K_LEN }).contiguous();

            std::vector<int64_t> records(NUM_FUNCTIONS, 0);
            fprintf(fptr, "Benchmarking L=%d M=%d K=%d N=%d \n", LAYERS, M_ROW, K_LEN, N_COL);
            for (int n = 0; n < names.size(); n++) {
                fprintf(fptr, names[n].c_str());
            }
            fprintf(fptr, "\n");
            for (int i = 0; i < NUM_RUNS; i++) {
                vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
                int j = 0;
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
                model.forward(X);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
                for (int j = 0; j < names.size(); j++) {
                    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timePoints[j + 1] - timePoints[j]).count();
                    records[j] += ns;
                    fprintf(fptr, "%lld \t", ns);
                }
                fprintf(fptr, "\n");
            }
            std::cout << "L=" << LAYERS << ", M=" << M_ROW << ", K=" << K_LEN << ", N=" << N_COL << std::endl;
            int64_t baseline = records[0];
            for (int n = 0; n < names.size(); n++) {
                float speedup = (double)baseline / (double)records[n];
                speedups.push_back(speedup);
                std::cout << names[n] << records[n] / 1000000 << "\t ms, speedup = " << std::fixed << std::setprecision(2) << speedup << std::endl;
                fprintf(fptr, "%lld\t", records[n] / 1000000);
            }
            std::cout << "\n" << std::endl;
        }
    }

    fclose(fptr);
    return 0;
}

int verify_FP32_GEMM() {
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        // {64, 128, 32, 0.9, 0.01},
        // {64, 32, 8, 0.75,0.01},
        {256, 1024,  4096, 0.50,0.05}, // Different Sparsity
     /* {256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.70,0.05},
        {256, 2048,  8192, 0.90,0.05},
        {256, 4096, 16384, 0.50,0.05},
        {256, 4096, 16384, 0.70,0.05},
        {256, 4096, 16384, 0.90,0.05},
        { 32, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.50,0.05},
       {1024, 2048,  8192, 0.50,0.05},
       {1024, 2048,  8192, 0.70,0.05},
       {1024, 2048,  8192, 0.90,0.05},*/
    };

    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {
        std::vector<float> Activation = initX<float>(M_ROW * K_LEN, 512);//Activation 
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed
        std::vector<int8_t> WeightAligned = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, true, false); //Weights aligned, not uniformed
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0)
        std::vector<float> Weight_FP32(Weight.begin(), Weight.end());
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness            
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<float> Y_CalB(M_ROW * N_COL, 0);
        std::vector<float> Y_CalC(M_ROW * N_COL, 0);
        std::vector<float> Y_CalD(M_ROW * N_COL, 0);
        // Convert vectors into *
        float* X_FP32 = Activation.data();
        int8_t* W_INT8 = Weight.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        float* Rc = Y_CalC.data();
        float* Rd = Y_CalD.data();
        //Ternary CSC arrays
        SparseFormat naiveTCSC = SparseFormat(W_INT8, K_LEN, N_COL);
        SparseFormat naiveTCSC_Aligned = SparseFormat(WeightAligned.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);
        int* w_col_start_pos = naiveTCSC.col_start_pos.data();
        int* w_col_start_neg = naiveTCSC.col_start_neg.data();
        int16_t* w_row_index_pos = naiveTCSC.row_index_pos.data();
        int16_t* w_row_index_neg = naiveTCSC.row_index_neg.data();
        MergedTCSC mergedTCSC = MergedTCSC(naiveTCSC, K_LEN, N_COL);
        // MergedTCSC_Group(SparseFormat naiveTCSC, int K, int N, int group_size, string group_method, bool interleaved) 
        MergedTCSC_Group mergedTCSC_G4_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 4, "min", true);
        MergedTCSC_Group mergedTCSC_G8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", true);
        MergedTCSC_Group mergedTCSC_G16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", true);
        MergedTCSC_Group mergedTCSC_G32_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", true);
        MergedTCSC_Group mergedTCSC_G64_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", true);
        MergedTCSC_Group mergedTCSC_G8C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", false);
        MergedTCSC_Group mergedTCSC_G16C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", false);
        MergedTCSC_Group mergedTCSC_G32C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false);
        MergedTCSC_Group mergedTCSC_G64C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false);
        MergedTCSC_Group mergedTCSC_G4_Mid = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 4, "mid", true);
        MergedTCSC_Group mergedTCSC_G4_Max = MergedTCSC_Group(naiveTCSC_Aligned, K_LEN, N_COL, 4, "max", true);
        MergedTCSC_Group uniformTCSC_G1 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 1, "uniform", true);
        MergedTCSC_Group uniformTCSC_G4 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 4, "uniform", true);
        MergedTCSC_Group uniformTCSC_G8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", true);
        MergedTCSC_Group uniformTCSC_G16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", true);
        MergedTCSC_Group uniformTCSC_G32 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", true);
        vector<int> G8C_index(mergedTCSC_G8C_Min.row_index.begin(), mergedTCSC_G8C_Min.row_index.end());
        vector<int> G16C_index(mergedTCSC_G16C_Min.row_index.begin(), mergedTCSC_G16C_Min.row_index.end());
        vector<int> G32C_index(mergedTCSC_G32C_Min.row_index.begin(), mergedTCSC_G32C_Min.row_index.end());
        vector<int> G64C_index(mergedTCSC_G64C_Min.row_index.begin(), mergedTCSC_G64C_Min.row_index.end());
        //Eigen version
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32, M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W1_eigen(Weight_FP32.data(), K_LEN, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> result_eigen(M_ROW, N_COL);

        //Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32, M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen(Weight_FP32.data(), K_LEN, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen(M_ROW, N_COL);
        Eigen::SparseMatrix<float> W_eigen_sparse = W_eigen.sparseView();

        torch::Tensor tensorX = torch::from_blob(Activation.data(), {M_ROW, K_LEN}, torch::kFloat32);
        torch::Tensor tensorW = torch::from_blob(Weight_FP32.data(), {K_LEN, N_COL}, torch::kFloat32);
        torch::Tensor tensorWcsc = tensorW.to_sparse_csc();
        torch::Tensor tensorWcsr = tensorW.to_sparse_csr();
        torch::Tensor tensorY    = torch::matmul(tensorX, tensorW);
        torch::Tensor tensorYcsr = torch::matmul(tensorX, tensorWcsr);
        torch::Tensor tensorYcsc = torch::matmul(tensorX, tensorWcsc);

        // ********* SiLU 
        std::vector<float> VA = initX<float>(K_LEN, 10);
        std::vector<float> VB = initX<float>(K_LEN, 10);
        std::vector<float> VA1(VA.begin(), VA.end());
        Naive_SiLU_Dot(K_LEN, VA1.data(), VB.data());
        std::vector<float> VA2(VA.begin(), VA.end());
        Naive_SiLU_Dot_Unroll_AVX2(K_LEN, VA2.data(), VB.data());
        if (compare_results(VA2.data(), VA1.data(), K_LEN, 0.2f)) {
            std::cout << "Passed! Naive_SiLU_Dot_Unroll_AVX2" << std::endl;
        }
        std::vector<float> VA3(VA.begin(), VA.end());
        Naive_SiLU_Dot_Unroll_AVX512(K_LEN, VA3.data(), VB.data());
        if (compare_results(VA3.data(), VA1.data(), K_LEN, 0.2f)) {
            std::cout << "Passed! Naive_SiLU_Dot_Unroll_AVX512" << std::endl;
        }

        // Eigen GEMM
        GEMM_CPU_FP32_colMajor_Direct_OpenMP(X_FP32, W_INT8, Ra, M_ROW, N_COL, K_LEN);
        Y_eigen = X_eigen * W_eigen;
        if (compare_results(Ra, Y_eigen.array().data(), M_ROW, N_COL)) {
            std::cout << "Passed! Eigen_Dense" << std::endl;
        }
        Y_eigen = X_eigen * W_eigen_sparse;       
        if (compare_results(Ra, Y_eigen.array().data(), M_ROW, N_COL)) {
            std::cout << "Passed! Eigen_Sparse" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_Direct_OpenMP(X_FP32, WeightUniform.data(), Y_CalB.data(), M_ROW, N_COL, K_LEN);
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Y_CalB.data(), Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Y_CalB.data(), Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP" << std::endl;
        }

        //*  GEMM functions 
        GEMM_CPU_FP32_colMajor_TCSC_Naive(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Naive" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_4x4_Unroll(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_4x4_Unroll" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_8x1_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_8x1_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_if_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_if_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMPi(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMPi" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMPj(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMPj" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Min.metadata.data(), mergedTCSC_G4_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16_Min.metadata.data(), mergedTCSC_G16_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG16_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG32_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG32_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX512_OpenMP(X_FP32, mergedTCSC_G32_Min.metadata.data(), mergedTCSC_G32_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX512_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_32xG1_AVX512_OpenMP(X_FP32, mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_32xG1_AVX512_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMid_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Mid.metadata.data(), mergedTCSC_G4_Mid.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMid_8xG4_AVX2_OpenMP" << std::endl;
        }

        // Aligned Weight
        GEMM_CPU_FP32_colMajor_Direct_OpenMP(X_FP32, WeightAligned.data(), Rc, M_ROW, N_COL, K_LEN);
        GEMM_CPU_FP32_colMajor_TCSC_Aligned_GroupMax_8xG4_AVX2_OpenMP(X_FP32, mergedTCSC_G4_Max.metadata.data(), mergedTCSC_G4_Max.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rc, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Aligned_GroupMax_8xG4_AVX2_OpenMP" << std::endl;
        }

        // Uniform Weight
        GEMM_CPU_FP32_colMajor_Direct_OpenMP(X_FP32, WeightUniform.data(), Rd, M_ROW, N_COL, K_LEN);
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG4_AVX2_OpenMP(X_FP32, uniformTCSC_G4.metadata[0], uniformTCSC_G4.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rd, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG4_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG8_AVX2_OpenMP(X_FP32, uniformTCSC_G8.metadata[0], uniformTCSC_G8.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rd, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG16_AVX2_OpenMP(X_FP32, uniformTCSC_G16.metadata[0], uniformTCSC_G16.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rd, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG16_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG32_AVX2_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rd, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG32_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX2_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rd, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG1_AVX512_OpenMP(X_FP32, uniformTCSC_G1.metadata[0], uniformTCSC_G1.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rd, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG1_AVX512_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX512_OpenMP(X_FP32, uniformTCSC_G32.metadata[0], uniformTCSC_G32.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rd, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX512_OpenMP" << std::endl;
        }

        GEMM_CPU_FP32_rowMajor_Direct_OpenMP(X_FP32, W_INT8, Ra, M_ROW, N_COL, K_LEN);
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG16_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG32_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG64_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP" << std::endl;
        } 
    }
    return 0;
}

int verify_FP32_GEMV() {
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        {  1, 1024,  4096, 0.50,0.05},
        /*{1, 1024,  4096, 0.60,0.05},
        {  1, 1024,  4096, 0.70,0.05},
        {  1, 1024,  4096, 0.80,0.05},
        {  1, 1024,  4096, 0.90,0.05},
        {  1, 2048,  8192, 0.50,0.05},
        {  1, 2048,  8192, 0.60,0.05},
        {  1, 2048,  8192, 0.70,0.05},
        {  1, 2048,  8192, 0.80,0.05},
        {  1, 2048,  8192, 0.90,0.05},
        {  1, 4096, 16384, 0.50,0.05},
        {  1, 4096, 16384, 0.60,0.05},
        {  1, 4096, 16384, 0.70,0.05},
        {  1, 4096, 16384, 0.80,0.05},
        {  1, 4096, 16384, 0.90,0.05},*/
    }; 
      
    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {
        std::vector<float> Activation = initX<float>(M_ROW * K_LEN, 512);//Activation 
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed
        std::vector<int8_t> WeightAligned = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, true, false); //Weights aligned, not uniformed
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0)
        std::vector<float> Weight_FP32(Weight.begin(), Weight.end());
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness            
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication
        // Convert vectors into *
        float* X_FP32 = Activation.data();
        int8_t* W_INT8 = Weight.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        //Ternary CSC arrays
        SparseFormat naiveTCSC = SparseFormat(W_INT8, K_LEN, N_COL);
        SparseFormat naiveTCSC_Aligned = SparseFormat(WeightAligned.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);
        int* w_col_start_pos = naiveTCSC.col_start_pos.data();
        int* w_col_start_neg = naiveTCSC.col_start_neg.data();
        int16_t* w_row_index_pos = naiveTCSC.row_index_pos.data();
        int16_t* w_row_index_neg = naiveTCSC.row_index_neg.data();
        MergedTCSC mergedTCSC = MergedTCSC(naiveTCSC, K_LEN, N_COL);
        // MergedTCSC_Group(SparseFormat naiveTCSC, int K, int N, int group_size, string group_method, bool interleaved) 
        MergedTCSC_Group mergedTCSC_G4_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 4, "min", true);
        MergedTCSC_Group mergedTCSC_G8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", true);
        MergedTCSC_Group mergedTCSC_G16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", true);
        MergedTCSC_Group mergedTCSC_G32_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", true);
        MergedTCSC_Group mergedTCSC_G64_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", true);
        MergedTCSC_Group mergedTCSC_G8C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", false);
        MergedTCSC_Group mergedTCSC_G16C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", false);
        MergedTCSC_Group mergedTCSC_G32C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false);
        MergedTCSC_Group mergedTCSC_G64C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false);
        MergedTCSC_Group mergedTCSC_G128C_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 128, "min", false);

        MergedTCSC_Group mergedTCSC_G16CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 16, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G32CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G64CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G128CS8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 128, "min", false, 8);
        MergedTCSC_Group mergedTCSC_G32CS16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 32, "min", false, 16);
        MergedTCSC_Group mergedTCSC_G64CS16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 64, "min", false, 16);
        MergedTCSC_Group mergedTCSC_G128CS16_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 128, "min", false, 16);

        MergedTCSC_Group mergedTCSC_G4_Mid = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 4, "mid", true);
        MergedTCSC_Group mergedTCSC_G4_Max = MergedTCSC_Group(naiveTCSC_Aligned, K_LEN, N_COL, 4, "max", true);
        MergedTCSC_Group uniformTCSC_G1 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 1, "uniform", true);
        MergedTCSC_Group uniformTCSC_G4 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 4, "uniform", true);
        MergedTCSC_Group uniformTCSC_G8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", true);
        MergedTCSC_Group uniformTCSC_G16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", true);
        MergedTCSC_Group uniformTCSC_G32 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", true);
        MergedTCSC_Group uniformTCSC_G64 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", true);
        MergedTCSC_Group uniformTCSC_G128 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", true);
        MergedTCSC_Group uniformTCSC_G8C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 8, "uniform", false);
        MergedTCSC_Group uniformTCSC_G16C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", false);
        MergedTCSC_Group uniformTCSC_G32C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", false);
        MergedTCSC_Group uniformTCSC_G64C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false);
        MergedTCSC_Group uniformTCSC_G128C = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", false);
        MergedTCSC_Group uniformTCSC_G16CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 16, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G32CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G64CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G128CS8 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", false, 8);
        MergedTCSC_Group uniformTCSC_G32CS16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 32, "uniform", false, 16);
        MergedTCSC_Group uniformTCSC_G64CS16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 64, "uniform", false, 16);
        MergedTCSC_Group uniformTCSC_G128CS16 = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, 128, "uniform", false, 16);
        vector<int> G8C_index(mergedTCSC_G8C_Min.row_index.begin(), mergedTCSC_G8C_Min.row_index.end());
        vector<int> G16C_index(mergedTCSC_G16C_Min.row_index.begin(), mergedTCSC_G16C_Min.row_index.end());
        vector<int> G32C_index(mergedTCSC_G32C_Min.row_index.begin(), mergedTCSC_G32C_Min.row_index.end());
        vector<int> G64C_index(mergedTCSC_G64C_Min.row_index.begin(), mergedTCSC_G64C_Min.row_index.end());
        vector<int> G128C_index(mergedTCSC_G128C_Min.row_index.begin(), mergedTCSC_G128C_Min.row_index.end());
        vector<int> G16CS8_index(mergedTCSC_G16CS8_Min.row_index.begin(), mergedTCSC_G16CS8_Min.row_index.end());
        vector<int> G32CS8_index(mergedTCSC_G32CS8_Min.row_index.begin(), mergedTCSC_G32CS8_Min.row_index.end());
        vector<int> G64CS8_index(mergedTCSC_G64CS8_Min.row_index.begin(), mergedTCSC_G64CS8_Min.row_index.end());
        vector<int> G128CS8_index(mergedTCSC_G128CS8_Min.row_index.begin(), mergedTCSC_G128CS8_Min.row_index.end());
        vector<int> G32CS16_index(mergedTCSC_G32CS16_Min.row_index.begin(), mergedTCSC_G32CS16_Min.row_index.end());
        vector<int> G64CS16_index(mergedTCSC_G64CS16_Min.row_index.begin(), mergedTCSC_G64CS16_Min.row_index.end());
        vector<int> G128CS16_index(mergedTCSC_G128CS16_Min.row_index.begin(), mergedTCSC_G128CS16_Min.row_index.end());
        vector<int> UG8C_index(uniformTCSC_G8C.row_index.begin(), uniformTCSC_G8C.row_index.end());
        vector<int> UG16C_index(uniformTCSC_G16C.row_index.begin(), uniformTCSC_G16C.row_index.end());
        vector<int> UG32C_index(uniformTCSC_G32C.row_index.begin(), uniformTCSC_G32C.row_index.end());
        vector<int> UG64C_index(uniformTCSC_G64C.row_index.begin(), uniformTCSC_G64C.row_index.end());
        vector<int> UG128C_index(uniformTCSC_G128C.row_index.begin(), uniformTCSC_G128C.row_index.end());
        vector<int> UG16CS8_index(uniformTCSC_G16CS8.row_index.begin(), uniformTCSC_G16CS8.row_index.end());
        vector<int> UG32CS8_index(uniformTCSC_G32CS8.row_index.begin(), uniformTCSC_G32CS8.row_index.end());
        vector<int> UG64CS8_index(uniformTCSC_G64CS8.row_index.begin(), uniformTCSC_G64CS8.row_index.end());
        vector<int> UG128CS8_index(uniformTCSC_G128CS8.row_index.begin(), uniformTCSC_G128CS8.row_index.end());
        vector<int> UG32CS16_index(uniformTCSC_G32CS16.row_index.begin(), uniformTCSC_G32CS16.row_index.end());
        vector<int> UG64CS16_index(uniformTCSC_G64CS16.row_index.begin(), uniformTCSC_G64CS16.row_index.end());
        vector<int> UG128CS16_index(uniformTCSC_G128CS16.row_index.begin(), uniformTCSC_G128CS16.row_index.end());


        // * GEMV
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32, M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W1_eigen(Weight_FP32.data(), K_LEN, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> result_eigen(M_ROW, N_COL);

        result_eigen = X_eigen * W1_eigen;
        GEMM_CPU_FP32_colMajor_TCSC_Naive(X_FP32, w_row_index_neg, w_col_start_neg, w_row_index_pos, w_col_start_pos, Ra, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, result_eigen.array().data(), M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_FP32_colMajor_TCSC_Naive" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(X_FP32, mergedTCSC_G8C_Min.metadata.data(), G8C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X_FP32, mergedTCSC_G16C_Min.metadata.data(), G16C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(X_FP32, mergedTCSC_G32C_Min.metadata.data(), G32C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP" << std::endl;
        }

        GEMM_CPU_FP32_rowMajor_Direct_OpenMP(X_FP32, W_INT8, Ra, M_ROW, N_COL, K_LEN);
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X_FP32, mergedTCSC_G64C_Min.metadata.data(), G64C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP(X_FP32, mergedTCSC_G128C_Min.metadata.data(), G128C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_AVX2_OpenMP(X_FP32, mergedTCSC_G64CS8_Min.metadata.data(), G64CS8_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD3_AVX2_OpenMP(X_FP32, mergedTCSC_G64CS8_Min.metadata.data(), G64CS8_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD3_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_AVX512_OpenMP(X_FP32, mergedTCSC_G128CS16_Min.metadata.data(), G128CS16_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_AVX512_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD3_AVX512_OpenMP(X_FP32, mergedTCSC_G128CS16_Min.metadata.data(), G128CS16_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD3_AVX512_OpenMP" << std::endl;
        }
        GEMM_CPU_FP32_rowMajor_Direct_OpenMP(X_FP32, WeightUniform.data(), Ra, M_ROW, N_COL, K_LEN);
        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP(X_FP32, uniformTCSC_G8C.metadata[0], UG8C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X_FP32, uniformTCSC_G64C.metadata[0], UG64C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_CS8_AVX2_OpenMP(X_FP32, uniformTCSC_G64CS8.metadata[0], UG64CS8_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_CS8_AVX2_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP(X_FP32, uniformTCSC_G128C.metadata[0], UG128C_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP" << std::endl;
        }
        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_CS16_AVX512_OpenMP(X_FP32, uniformTCSC_G128CS16.metadata[0], UG128CS16_index.data(), Rb, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_CS16_AVX512_OpenMP" << std::endl;
        }
    }
    return 0;
}

int verify_INT8_GEMM() {
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        // {64, 128, 32, 0.9, 0.01},
        // {64, 32, 8, 0.75,0.01},
        {256, 1024,  4096, 0.50,0.05}, // Different Sparsity
     /* {256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.70,0.05},
        {256, 2048,  8192, 0.90,0.05},
        {256, 4096, 16384, 0.50,0.05},
        {256, 4096, 16384, 0.70,0.05},
        {256, 4096, 16384, 0.90,0.05},
        { 32, 2048,  8192, 0.50,0.05},
        {256, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.50,0.05},
       {1024, 2048,  8192, 0.50,0.05},
       {1024, 2048,  8192, 0.70,0.05},
       {1024, 2048,  8192, 0.90,0.05}, */
    }; 

    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {

        std::vector<int8_t> Activation = initX<int8_t>(M_ROW * K_LEN, 10);//Activation 
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0)
        SparseFormat naiveTCSC = SparseFormat(Weight.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);

        std::vector<int8_t> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness
        std::vector<int8_t> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<int8_t> Y_CalB(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<float> X_FP = initX<float>(M_ROW * K_LEN, 10);//Activation
        std::vector<float> Y_FP(M_ROW * N_COL, 0);  //Result of the multiplication
        // Convert vectors into * 
        int8_t* Ra = Y_Ref.data();
        int8_t* Rb = Y_Cal.data();
        int8_t* Rc = Y_CalB.data();
        float* X_FP32 = X_FP.data();
        float* Y_FP32 = Y_FP.data();
        int Gourp_Size = 4;
        MergedTCSC_Group mergedTCSC = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, Gourp_Size, "min", false);
        MergedTCSC_Group uniformTCSC = MergedTCSC_Group(naiveTCSC_Uniform, K_LEN, N_COL, Gourp_Size, "uniform", false);


        GEMM_CPU_colMajor_Direct_OpenMP(Activation.data(), WeightUniform.data(), Ra, M_ROW, N_COL, K_LEN);
        GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP(Activation.data(), uniformTCSC.metadata[0], uniformTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP" << std::endl;
        }
        // GEMM_CPU_colMajor_Direct_OpenMP(Activation.data(), WeightUniform.data(), Ra, M_ROW, N_COL, K_LEN);
        GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX2_OpenMP(Activation.data(), uniformTCSC.metadata[0], uniformTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Ra, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX2_OpenMP" << std::endl;
        }
        GEMM_CPU_colMajor_Direct_OpenMP(Activation.data(), Weight.data(), Rc, M_ROW, N_COL, K_LEN);
        GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_64xG4_AVX2_OpenMP(Activation.data(), mergedTCSC.metadata.data(), mergedTCSC.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
        if (compare_results(Rc, Rb, M_ROW, N_COL)) {
            std::cout << "Passed! GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_64xG4_AVX2_OpenMP" << std::endl;
        }
    }
    
    return 0;
}


int main() {
    verify_FP32_GEMM();
    verify_FP32_GEMV();
    verify_INT8_GEMM();
    benchmark_FP32_GEMM_General_Optimizations();
    benchmark_FP32_Matrix_Level();
    benchmark_FP32_Layer_Level();
    benchmark_FP32_Llama_Model();
    std::cin.get();
    return 0;
}