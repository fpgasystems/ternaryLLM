#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

#include "GEMM_CPU_INT8.hpp"
#include "GEMM_CPU_FP32.hpp"
#include "TCSC.hpp"
#include "initData.hpp"
#include <torch/torch.h>
using namespace std;


class LlamaLayer {
public:   
    torch::nn::Linear q_proj = nullptr;
    torch::nn::Linear k_proj = nullptr;
    torch::nn::Linear v_proj = nullptr;
    torch::nn::MultiheadAttention MHA = nullptr;
    torch::nn::Linear o_proj = nullptr;
    torch::nn::Linear up_proj = nullptr;
    torch::nn::Linear gate_proj = nullptr;
    torch::nn::Linear down_proj = nullptr;
    int QHEADS;
    int KVHEADS;
    int HEAD_SIZE;
    //int EMB_SIZE;
    //int IMM_SIZE;
    LlamaLayer(int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE) {        
        this->QHEADS = QHEADS;
        this->KVHEADS = KVHEADS;
        this->HEAD_SIZE = EMBEDDING_SIZE / QHEADS;
        int KV_SIZE = this->HEAD_SIZE * KVHEADS;
        this->q_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->k_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->v_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->MHA = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        this->o_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->up_proj = torch::nn::Linear(EMBEDDING_SIZE, INTERMEDIATE_SIZE);
        this->gate_proj = torch::nn::Linear(EMBEDDING_SIZE, INTERMEDIATE_SIZE);
        this->down_proj = torch::nn::Linear(INTERMEDIATE_SIZE, EMBEDDING_SIZE);
    }

    torch::Tensor forward(torch::Tensor X) {
        torch::Tensor X_Residual = X;
        torch::Tensor Query = this->q_proj(X);
        torch::Tensor Key = this->k_proj(X).repeat_interleave(this->QHEADS/this->KVHEADS, 2);
        torch::Tensor Value = this->v_proj(X).repeat_interleave(this->QHEADS / this->KVHEADS, 2);
        // torch::nn::functional::multi_head_attention_forward(Query, Key, Value, torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        auto Output = this->MHA->forward(Query, Key, Value);
        torch::Tensor Y = this->o_proj(std::get<0>(Output));
        X = Y + X_Residual;
        X_Residual = X;
        torch::Tensor Up = this->up_proj(X);
        torch::Tensor Gate = this->gate_proj(X);
        Gate = torch::nn::functional::silu(Gate);
        Gate = torch::mul(Gate, Up);
        Y = this->down_proj(Gate);
        X = Y + X_Residual;
        return X;
    }
};

class LlamaModel {
public:
    vector<LlamaLayer> layers;

    LlamaModel(int LAYERS, int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE) {
        for (int i = 0; i < LAYERS; i++) {
            auto layer = LlamaLayer(QHEADS, KVHEADS, EMBEDDING_SIZE, INTERMEDIATE_SIZE);
            this->layers.push_back(layer);
        }
    }

    torch::Tensor forward(torch::Tensor X) {
        torch::Tensor Y = X;
        for (int i = 0; i < this->layers.size(); i++) {
            Y = this->layers[i].forward(Y);
        }
        return Y;
    }
};


template<typename T>
class TernaryMLP {
public:
    TernaryMLP() {}
    void forward() {}
};

template<>
class TernaryMLP<int32_t> {
public:
    vector<int> WU_metadata;
    vector<int> WG_metadata;
    vector<int> WD_metadata;
    vector<int32_t> WU_rowindex;
    vector<int32_t> WG_rowindex;
    vector<int32_t> WD_rowindex;
    int Group;
    bool Uniform;
    bool AVX512;
    bool Generation;
    TernaryMLP() {}
    TernaryMLP(int K_LEN, int N_COL, float Sparsity, float Variation, bool Uniform, bool AVX512, bool Generation) {
        this->Uniform = Uniform;
        this->AVX512 = AVX512;
        this->Generation = Generation;
        // Get threads and calculate max group number
        int threads = std::thread::hardware_concurrency();
        if (threads < 1)
            threads = 16;
        int max_group = std::min(N_COL, K_LEN) / threads;
        // Determine appropriate group size
        if (Generation) {
            if (max_group >= 64) {
                this->Group = 64;
            }
            else {
                this->Group = max_group;
            }
        }
        else {
            if (AVX512) {
                if (max_group >= 32) {
                    this->Group = 32;
                }
                else {
                    this->Group = max_group;
                }

            }
            else {
                if (max_group >= 16) {
                    this->Group = 16;
                }
                else {
                    this->Group = max_group;
                }
            }
        }
        // Initialize the weights
        std::string mode = "min";
        if (Uniform) {
            mode = "uniform";
        }
        std::vector<int8_t> WeightU = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, Uniform); //Weights not aligned, not uniformed
        std::vector<int8_t> WeightG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, Uniform); // Gate proj
        std::vector<int8_t> WeightD = sparseWeight<int8_t>(N_COL, K_LEN, Sparsity, Variation, false, Uniform); // Down proj
        SparseFormat naiveTCSCU = SparseFormat(WeightU.data(), K_LEN, N_COL);
        SparseFormat naiveTCSCG = SparseFormat(WeightG.data(), K_LEN, N_COL);
        SparseFormat naiveTCSCD = SparseFormat(WeightD.data(), N_COL, K_LEN);
        MergedTCSC_Group TCSCU = MergedTCSC_Group(naiveTCSCU, K_LEN, N_COL, this->Group, mode, false);
        MergedTCSC_Group TCSCG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, this->Group, mode, false);
        MergedTCSC_Group TCSCD = MergedTCSC_Group(naiveTCSCD, N_COL, K_LEN, this->Group, mode, false);
        this->WU_metadata = vector<int32_t>(TCSCU.metadata.begin(), TCSCU.metadata.end());
        this->WG_metadata = vector<int32_t>(TCSCG.metadata.begin(), TCSCG.metadata.end());
        this->WD_metadata = vector<int32_t>(TCSCD.metadata.begin(), TCSCD.metadata.end());
        this->WU_rowindex = vector<int32_t>(TCSCU.row_index.begin(), TCSCU.row_index.end());
        this->WG_rowindex = vector<int32_t>(TCSCG.row_index.begin(), TCSCG.row_index.end());
        this->WD_rowindex = vector<int32_t>(TCSCD.row_index.begin(), TCSCD.row_index.end());
    }

    void forward(float* X, float* Ya, float* Yb, int M_ROW, int N_COL, int K_LEN) {
        if (this->Uniform) {
            if (M_ROW == 1) {
                if (this->AVX512) {
                    switch (this->Group) {
                    case 16:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 32:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 64:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 128:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    }
                }
                else {
                    switch (this->Group) {
                    case 8:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 16:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 32:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 64:
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                        GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    }
                }
            }
        }
        else {
            if (M_ROW == 1) {
                if (this->AVX512) {
                    switch (this->Group) {
                    case 16:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 32:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 64:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 128:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    }
                }
                else {
                    switch (this->Group) {
                    case 8:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 16:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 32:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    case 64:
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, N_COL, K_LEN);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, N_COL, K_LEN);
                        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                        GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, K_LEN, N_COL); break;
                    }
                }
            }
        }
    }
};

template<>
class TernaryMLP<int16_t> {
public:
    vector<int> WU_metadata;
    vector<int> WG_metadata;
    vector<int> WD_metadata;
    vector<int16_t> WU_rowindex;
    vector<int16_t> WG_rowindex;
    vector<int16_t> WD_rowindex;
    int Group;
    bool Uniform;
    bool AVX512;
    bool Generation;
    TernaryMLP() {}
    TernaryMLP(int K_LEN, int N_COL, float Sparsity, float Variation, bool Uniform, bool AVX512, bool Generation) {
        this->Uniform = Uniform;
        this->AVX512 = AVX512;
        this->Generation = Generation;
        // Get threads and calculate max group number
        int threads = std::thread::hardware_concurrency();
        if (threads < 1)
            threads = 16;
        int max_group = std::min(N_COL, K_LEN) / threads;
        // Determine appropriate group size
        if (AVX512) {
            if (max_group >= 32) {
                this->Group = 32;
            }
            else {
                this->Group = max_group;
            }

        }
        else {
            if (max_group >= 16) {
                this->Group = 16;
            }
            else {
                this->Group = max_group;
            }
        }
        // Initialize the weights
        std::string mode = "min";
        if (Uniform) {
            mode = "uniform";
        }
        std::vector<int8_t> WeightU = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, Uniform); //Weights not aligned, not uniformed
        std::vector<int8_t> WeightG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, Uniform); // Gate proj
        std::vector<int8_t> WeightD = sparseWeight<int8_t>(N_COL, K_LEN, Sparsity, Variation, false, Uniform); // Down proj
        SparseFormat naiveTCSCU = SparseFormat(WeightU.data(), K_LEN, N_COL);
        SparseFormat naiveTCSCG = SparseFormat(WeightG.data(), K_LEN, N_COL);
        SparseFormat naiveTCSCD = SparseFormat(WeightD.data(), N_COL, K_LEN);
        MergedTCSC_Group TCSCU = MergedTCSC_Group(naiveTCSCU, K_LEN, N_COL, this->Group, mode, false);
        MergedTCSC_Group TCSCG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, this->Group, mode, false);
        MergedTCSC_Group TCSCD = MergedTCSC_Group(naiveTCSCD, N_COL, K_LEN, this->Group, mode, false);
        // Must use Data Copy, otherwise the data will be wrong!!!!!!!!!!!
        //this->WU_metadata = TCSCU.metadata;
        //this->WG_metadata = TCSCG.metadata;
        //this->WD_metadata = TCSCD.metadata;
        //this->WU_rowindex = TCSCU.row_index;
        //this->WG_rowindex = TCSCG.row_index;
        //this->WD_rowindex = TCSCD.row_index;
        this->WU_metadata = vector<int32_t>(TCSCU.metadata.begin(), TCSCU.metadata.end());
        this->WG_metadata = vector<int32_t>(TCSCG.metadata.begin(), TCSCG.metadata.end());
        this->WD_metadata = vector<int32_t>(TCSCD.metadata.begin(), TCSCD.metadata.end());
        this->WU_rowindex = vector<int16_t>(TCSCU.row_index.begin(), TCSCU.row_index.end());
        this->WG_rowindex = vector<int16_t>(TCSCG.row_index.begin(), TCSCG.row_index.end());
        this->WD_rowindex = vector<int16_t>(TCSCD.row_index.begin(), TCSCD.row_index.end());
    }

    void forward(float* X, float* Ya, float* Yb, int M_ROW, int N_COL, int K_LEN) {
        if (this->Uniform) {
            if (M_ROW > 1) {
                if (M_ROW % 32 == 0) {
                    if (this->AVX512) {
                        switch (this->Group) {
                        case 4:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 8:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 16:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 32:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX512_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX512_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX512(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX512_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        }
                    }
                    else {
                        switch (this->Group) {
                        case 4:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 8:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 16:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 32:
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX2_OpenMP(X, this->WU_metadata[0], this->WU_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX2_OpenMP(X, this->WG_metadata[0], this->WG_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Ya, Yb);
                            GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX2_OpenMP(Ya, this->WD_metadata[0], this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        }
                    }
                }
            }
        }
        else {
            if (M_ROW > 1) {
                if (M_ROW % 32 == 0) {
                    if (this->AVX512) {
                        switch (this->Group) {
                        case 4:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 8:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 16:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 32:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX512_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX512_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX512_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        }
                    }
                    else {
                        switch (this->Group) {
                        case 4:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 8:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 16:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        case 32:
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX2_OpenMP(X, this->WU_metadata.data(), this->WU_rowindex.data(), Ya, M_ROW, N_COL, K_LEN);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX2_OpenMP(X, this->WG_metadata.data(), this->WG_rowindex.data(), Yb, M_ROW, N_COL, K_LEN);
                            Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb, Ya);
                            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX2_OpenMP(Yb, this->WD_metadata.data(), this->WD_rowindex.data(), X, M_ROW, K_LEN, N_COL); break;
                        }
                    }
                }
            }
        }
    }

};


template <typename T>
class TernaryLlamaLayer {
public:
    torch::nn::Linear q_proj = nullptr;
    torch::nn::Linear k_proj = nullptr;
    torch::nn::Linear v_proj = nullptr;
    torch::nn::MultiheadAttention MHA = nullptr;
    torch::nn::Linear o_proj = nullptr;
    TernaryMLP<T> MLP;
    int QHEADS;
    int KVHEADS;
    int HEAD_SIZE;
    //int EMB_SIZE;
    int IMM_SIZE;
    TernaryLlamaLayer(int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE, float Sparsity, float Variation, bool Uniform, bool AVX512, bool Generation) {
        this->QHEADS = QHEADS;
        this->KVHEADS = KVHEADS;
        this->HEAD_SIZE = EMBEDDING_SIZE / QHEADS;
        this->IMM_SIZE = INTERMEDIATE_SIZE;
        int KV_SIZE = this->HEAD_SIZE * KVHEADS;
        this->q_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->k_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->v_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->MHA = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        this->o_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->MLP = TernaryMLP<T>(EMBEDDING_SIZE, INTERMEDIATE_SIZE, Sparsity, Variation, Uniform, AVX512, Generation);
    }

    torch::Tensor forward(torch::Tensor X, float * XT, float * Ya, float * Yb) {
        torch::Tensor X_Residual = X.contiguous();
        torch::Tensor Query = this->q_proj(X);
        torch::Tensor Key = this->k_proj(X).repeat_interleave(this->QHEADS / this->KVHEADS, 2);
        torch::Tensor Value = this->v_proj(X).repeat_interleave(this->QHEADS / this->KVHEADS, 2);
        // torch::nn::functional::multi_head_attention_forward(Query, Key, Value, torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        auto Output = this->MHA->forward(Query, Key, Value);
        torch::Tensor Y = this->o_proj(std::get<0>(Output));
        X = Y + X_Residual;
        X_Residual = X;
        FastMatrixTranspose(X.data_ptr<float>(), XT, X.sizes()[0] * X.sizes()[1], X.sizes()[2]); // Use dimension sizes after permute
        this->MLP.forward(XT, Ya, Yb, X.sizes()[0] * X.sizes()[1], this->IMM_SIZE, X.sizes()[2]);
        FastMatrixTranspose(XT, X.data_ptr<float>(), X.sizes()[2], X.sizes()[0] * X.sizes()[1]);
        X = X + X_Residual;
        return X;
    }
};

template <typename T>
class TernaryLlamaModel {
public:
    vector<TernaryLlamaLayer<T>> layers;
    int INTERMEDIATE_SIZE;

    TernaryLlamaModel(int LAYERS, int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE, float Sparsity, float Variation, bool Uniform, bool AVX512, bool Generation) {
        this->INTERMEDIATE_SIZE = INTERMEDIATE_SIZE;
        for (int i = 0; i < LAYERS; i++) {
            auto layer = TernaryLlamaLayer<T>(QHEADS, KVHEADS, EMBEDDING_SIZE, INTERMEDIATE_SIZE, Sparsity, Variation, Uniform, AVX512, Generation);
            this->layers.push_back(layer);
        }
    }

    torch::Tensor forward(torch::Tensor X) {
        int vlen = X.sizes()[0] * X.sizes()[1] * this->INTERMEDIATE_SIZE;
        vector<float> XT(X.sizes()[0] * X.sizes()[1] * X.sizes()[2], 0.0);
        vector<float> YA(vlen, 0.0);
        vector<float> YB(vlen, 0.0);
        for (int i = 0; i < this->layers.size(); i++) {
            this->layers[i].forward(X, XT.data(), YA.data(), YB.data());
        }
        return X;
    }
};
