
// 1. Allocate Coyote Thread
// 2. Allocate through getMem() 3 Locations X,W,Y
// 3. Put values into X,W,Y
// 4. Create sg_entry.sync for LOCAL_OFFLOAD and wait until it is done!
// 5. set Registers setCSR() (only start not) and wait until it is completed
// 6. set start signal
// 7. wait until done_signal using getCSR
// 8. Read out Y
// 9  free Memory

#include <any>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <cassert>

// Coyote-specific includes
#include "cThread.hpp"
// using namespace fpga;  // for master/branch
using namespace coyote;

// Defined in Config
#define S_SLICE 64
#define N_Slice 32
#define K_SLICE 128
#define AXI_DATA_LEN 64
#define UNROLL_M 4 
#define BYTE_X_VAL 1

// Default vFPGA to assign cThreads to
#define DEFAULT_VFPGA_ID 0

// Registers for axi_ctrl
#define INPUT_M 0
#define INPUT_N 1
#define INPUT_K 2
#define INPUT_NZ_KSLICE 3
#define START 4
#define ADDR_X 5
#define ADDR_W 6
#define ADDR_Y 7
#define DONE 8
#define EXPECTED_BEATS 9
#define CLOCK_COUNTER 10

// Define 5s timeouts for Coyote operations
// const std::chrono::seconds TIMEOUT(120);

// Generates Activation X
void generateX(int8_t *X, int M, int K)
{
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < K; ++j)
        {
            if (j % 2 == 0)
            {

                X[i * K + j] = 5;
            }
            else
            {

                X[i * K + j] = 1;
            }
        }
    }
}

// Generates Grouped Merged Uniformal Weights
void generateWUniformal(uint8_t *W, int S, int entries, int N)
{

    int times = N / N_Slice;
    int total_rows = times * entries;

    for (size_t i = 0; i < total_rows; ++i)
    {
        for (size_t j = 0; j < S; ++j)
        {

            if (j % 2 == 0)
            {

                W[i * S + j] = 0;
            }
            else
            {

                W[i * S + j] = 1;
            }
        }
    }
}

//Verify by comparing the output matrix to the naive implementation's results
void assertOutput(const int16_t *arr1, const int16_t *arr2, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (arr1[i] != arr2[i])
        {
            std::cerr << "Mismatch at index " << i << ": "
                      << arr1[i] << " != " << arr2[i] << std::endl;
            assert(false && "Arrays differ!");
        }
    }
}

// Naiive ternary GEMM
int16_t *naiveGEMM(int M, int N, int S, int K, int entries, const int8_t *X, const uint8_t *W, int Nz_K_Slice)
{
    int S_2 = S / 2;
    int times = N / S_2;
    int slice_weights;

    int16_t *Y = (int16_t *)malloc(M * N * sizeof(int16_t));
    if (Y == NULL)
    {
        throw std::runtime_error("Could not allocate memory for Y; exiting...");
    }

    for (int m = 0; m < M; ++m)
    {
        // Variable for the Kslice logic
        slice_weights = 0;
        for (int n = 0; n < N; ++n)
        {

            // calculate column group and offset(inside that group)
            int col_group = n / S_2;
            int col_offset = n % S_2;

            for (int e = 0; e < entries; ++e)
            {
                //calculate column group base
                int base = col_group * entries + e;
                
                // check if we have processed all indices of this KSLICE
                if (entries % (Nz_K_Slice - 1) == 0)
                {
                    slice_weights += 1;
                }
                
                // calculate weight index
                int w_index = base * S + (2 * col_offset);

                // calculate index for activation matrix
                uint8_t pos = W[w_index] + slice_weights * K_SLICE;
                uint8_t neg = W[w_index + 1] + slice_weights * K_SLICE;
                
                // reading out pos and neg activation values
                int16_t x_pos = X[m * K + pos];
                int16_t x_neg = X[m * K + neg];
                
                // perform addition and substraction
                Y[m * N + n] += (x_pos - x_neg);
            }
        }
    }

    return Y;
}

// Run individual ternary GEMM
void run_test(cThread<std::any> *coyote_thread, int M, int N, int K, double sparsity)
{

    std::cout << "\n=== Running testcase: Seq_len=" << M << ", intermediate_size=" << N << ", hidden_size=" << K << ", sparsity=" << sparsity << " ===" << std::endl;

    // DEBUG:
    // std::cout << "Get dataFSM_state   = " << coyote_thread->getCSR(static_cast<uint32_t>(21))<< std::endl;
    // std::cout << "Get START = " << coyote_thread->getCSR(static_cast<uint32_t>(START)) << std::endl;
    // std::cout << "Get DONE = " << coyote_thread->getCSR(static_cast<uint32_t>(DONE)) << std::endl;



    // Calculate entries and per K_SLICE
    int entries = static_cast<int>(((1 - sparsity) * K) / 2);
    int entries_Kslice = static_cast<int>(entries / (K / K_SLICE));
    int expected_beats = static_cast<int>(((UNROLL_M * BYTE_X_VAL * K)/ AXI_DATA_LEN) - 1);

    // For W-Index
    int times_W = N / N_Slice;
    int total_row_W = times_W * entries;

    std::cout << "Starting" << std::endl;
    // Size for allocating Memory
    u_int size_X = (uint)(M * K) * (uint)sizeof(int8_t);
    u_int size_W = (uint)(total_row_W * S_SLICE) * (uint)sizeof(uint8_t);
    u_int size_Y = (uint)(M * N) * (uint)sizeof(int16_t);

    // Outputs results from above
    // std::cout << "Computed sizes:" << std::endl;
    // std::cout << "  size_X = " << size_X << " bytes" << std::endl;
    // std::cout << "  size_W = " << size_W << " bytes" << std::endl;
    // std::cout << "  size_Y = " << size_Y << " bytes" << std::endl;
    // std::cout << "  total_row_W = " << total_row_W << std::endl;
    std::cout << "  entries = " << entries << std::endl;
    std::cout << "  entries_Kslice = " << entries_Kslice << std::endl;
    // std::cout << "  times_W = " << times_W << std::endl;
    // std::cout << "  expected_beats = " << expected_beats << std::endl;

    // Allocate Memory for X,W,Y
    int8_t *X = (int8_t *)coyote_thread->getMem({CoyoteAlloc::HPF, size_X});
    uint8_t *W = (uint8_t *)coyote_thread->getMem({CoyoteAlloc::HPF, size_W});
    int16_t *Y = (int16_t *)coyote_thread->getMem({CoyoteAlloc::HPF, size_Y});

    if (!X)
    {
        throw std::runtime_error("Failed to allocate memory for X");
    }
    if (!Y)
    {
        throw std::runtime_error("Failed to allocate memory for Y");
    }
    if (!W)
    {
        throw std::runtime_error("Failed to allocate memory for W");
    }
    // std::cout << "getMem completed" << std::endl;

    memset(Y, 0, size_Y);
    // std::cout << " Zero-out Y" << std::endl;

    generateX(X, M, K);
    generateWUniformal(W, S_SLICE, entries, N);
    // std::cout << "Created X and W" << std::endl;

    // Create sg_entries
    sgEntry sg_offload_X;
    sgEntry sg_offload_W;
    sgEntry sg_Y;

    // LOCAL_OFFLOAD
    sg_offload_X.sync.addr = X;
    sg_offload_X.sync.size = size_X;

    sg_offload_W.sync.addr = W;
    sg_offload_W.sync.size = size_W;

    sg_Y.sync.addr = Y;
    sg_Y.sync.size = size_Y;

    // std::cout << "LOCAL_OFFLOAD started" << std::endl;

    // std::cout << "Invoking X offload..." << std::endl;
    coyote_thread->invoke(CoyoteOper::LOCAL_OFFLOAD, &sg_offload_X);
    // // std::cout << "X offload done" << std::endl;

    // std::cout << "Invoking W offload..." << std::endl;
    coyote_thread->invoke(CoyoteOper::LOCAL_OFFLOAD, &sg_offload_W);
    // std::cout << "W offload done" << std::endl;

    // std::cout << "Invoking Y offload..." << std::endl;
    coyote_thread->invoke(CoyoteOper::LOCAL_OFFLOAD, &sg_Y);
    // std::cout << "Y offload done" << std::endl;

    // 1GB/s transfer
    sleep(2);

    bool stuck = false;

    // std::cout << "LOCAL_OFFLOAD completed" << std::endl;

    // // Set registers
    // std::cout << "Setting CSR: INPUT_M = 0x" << std::hex << (INPUT_M * 8) << ", Value = " << std::dec << M << std::endl;
    coyote_thread->setCSR((uint64_t)M, INPUT_M);
    // std::cout << "Get INPUT_M = " << coyote_thread->getCSR(static_cast<uint32_t>(INPUT_M)) << std::endl;

    // std::cout << "Setting CSR: INPUT_K = 0x" << std::hex << (INPUT_K * 8) << ", Value = " << std::dec << K << std::endl;
    coyote_thread->setCSR((uint64_t)K, INPUT_K);
    // std::cout << "Get INPUT_K = " << coyote_thread->getCSR(static_cast<uint32_t>(INPUT_K)) << std::endl;

    // std::cout << "Setting CSR: INPUT_N = 0x" << std::hex << (INPUT_N * 8) << ", Value = " << std::dec << N << std::endl;
    coyote_thread->setCSR((uint64_t)N, INPUT_N);
    // std::cout << "Get INPUT_N = " << coyote_thread->getCSR(static_cast<uint32_t>(INPUT_N)) << std::endl;

    // std::cout << "Setting CSR: INPUT_NZ_KSLICE  = 0x" << std::hex << (INPUT_NZ_KSLICE * 8) << ", Value = " << std::dec << entries_Kslice << std::endl;
    coyote_thread->setCSR((uint64_t)entries_Kslice, INPUT_NZ_KSLICE);
    //std::cout << "Get INPUT_NZ_KSLICE  = " << coyote_thread->getCSR(static_cast<uint32_t>(INPUT_NZ_KSLICE)) << std::endl;

    //std::cout << "Setting CSR: EXPECTED_BEATS  = 0x" << std::hex << (EXPECTED_BEATS * 8) << ", Value = " << std::dec << expected_beats << std::endl;
    coyote_thread->setCSR((uint64_t)expected_beats, EXPECTED_BEATS);
    //std::cout << "Get EXPECTED_BEATS  = " << coyote_thread->getCSR(static_cast<uint32_t>(EXPECTED_BEATS)) << std::endl;

    //std::cout << "Setting CSR: ADDR_X = 0x" << std::hex << (ADDR_X * 8) << ", Value = " << X << std::endl;
    coyote_thread->setCSR((uint64_t)X, ADDR_X);
    //std::cout << "Get ADDR_X = " << coyote_thread->getCSR(static_cast<uint32_t>(ADDR_X)) << std::endl;

    //std::cout << "Setting CSR: ADDR_W = 0x" << std::hex << (ADDR_W * 8) << ", Value = " << static_cast<void *>(W) << std::endl; // Because 1 Byte is treated as CHAR and we never set the terminate character
    coyote_thread->setCSR((uint64_t)W, ADDR_W);
    //std::cout << "Get ADDR_W = " << coyote_thread->getCSR(static_cast<uint32_t>(ADDR_W)) << std::endl;

    //std::cout << "Setting CSR: ADDR_Y = 0x" << std::hex << (ADDR_Y * 8) << ", Value = " << Y << std::endl;
    coyote_thread->setCSR((uint64_t)Y, ADDR_Y);
    //std::cout << "Get ADDR_Y = " << coyote_thread->getCSR(static_cast<uint32_t>(ADDR_Y)) << std::endl;

    //std::cout << "Setting CSR: START = 0x" << std::hex << (START * 8) << ", Value = " << 1 << std::endl;
    // Start measuring
    auto t_start = std::chrono::high_resolution_clock::now();
    coyote_thread->setCSR(1, START);
    //std::cout << "Get START = " << coyote_thread->getCSR(static_cast<uint32_t>(START)) << std::endl;

    // wait until Done
    auto start_done = std::chrono::high_resolution_clock::now();
    const auto timeout = std::chrono::seconds(20);
    while (coyote_thread->getCSR(static_cast<uint32_t>(DONE)) != 1)
    {
        auto now = std::chrono::high_resolution_clock::now();
        if (now - start_done > timeout)
        {
            stuck = true;
            break;

        }
    };

    // Calculate duration
    auto t_end = std::chrono::high_resolution_clock::now();
    int64_t duration_ns = std::chrono::duration<int64_t, std::nano>(t_end - t_start).count();
    std::cout << std::dec;
    std::cout << "Test finished in " << duration_ns << " ns" << std::endl;
    std::cout << "Cycles = " << coyote_thread->getCSR(static_cast<uint32_t>(CLOCK_COUNTER)) << std::endl;


    std::cout << std::dec;

    // DEBUG
    // if (stuck)
    // {
    //     for (int i = 0; i < 3; i++)
    //     {

    //         std:cout << "Iteration" << i << std::endl;

    //         std::cout << "Get cnt_M           = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(9))
    //                   << std::endl;

    //         std::cout << "Get cnt_N           = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(10))
    //                   << std::endl;

    //         std::cout << "Get cnt_K           = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(11))
    //                   << std::endl;

    //         std::cout << "Get cnt_entries     = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(12))
    //                   << std::endl;

    //         std::cout << "Get cnt_beats_W     = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(13))
    //                   << std::endl;

    //         std::cout << "Get cnt_unroll_Y    = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(14))
    //                   << std::endl;

    //         std::cout << "Get cnt_unroll_rows = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(15))
    //                   << std::endl;

    //         std::cout << "Get k_c             = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(16))
    //                   << std::endl;

    //         std::cout << "Get cnt_s           = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(17))
    //                   << std::endl;

    //         std::cout << "Get cnt_row         = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(18))
    //                   << std::endl;

    //         std::cout << "Get cnt_N_write     = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(19))
    //                   << std::endl;

    //         std::cout << "Get waitCounter     = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(20))
    //                   << std::endl;

    //         std::cout << "Get dataFSM_state   = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(21))
    //                   << std::endl;

    //         std::cout << "Get cnt_beats_y     = "
    //                   << coyote_thread->getCSR(static_cast<uint32_t>(22))
    //                   << std::endl;

    //         sleep(2);

    //         if (i == 2)
    //         {
    //             return;
    //         }
    //     }
    // }

    // RESET
    coyote_thread->setCSR(0, START);
    
    std:cout << "Reset Hardware" << std::endl;

    // Debug:
    // std::cout << "Get START = " << coyote_thread->getCSR(static_cast<uint32_t>(START)) << std::endl;
    // std::cout << "Get DONE = " << coyote_thread->getCSR(static_cast<uint32_t>(DONE)) << std::endl;
    // std::cout << "Done Signal is high" << std::endl;

    // LOCAL_SYNC
    sg_Y.sync.addr = Y;
    sg_Y.sync.size = size_Y;

    coyote_thread->invoke(CoyoteOper::LOCAL_SYNC, &sg_Y);
    sleep(2);

    // For verification:
    // int16_t *Y_Naive = naiveGEMM(M, N, S_SLICE, K, entries, X, W, entries_Kslice);
    // std::cout << std::dec;

    //assertOutput(Y_Naive, Y, M * N);

    // // Output Naive Y :
    // std::cout << "Output Naive Y" << std::endl;
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         std::cout << Y_Naive[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // Output Y :
    // std::cout << "Output calculated Y" << std::endl;
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         std::cout << Y[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Free Memory
    coyote_thread->freeMem(X);
    coyote_thread->freeMem(W);
    coyote_thread->freeMem(Y);
    
    std::cout << "DONE" << std::endl;

}

// Evaluation benchmark: Up and Down Projection
void run_benchmark(cThread<std::any> *coyote_thread){

    std::cout << "Start benchmark" << std::endl;

     // Define parameters
    const int seq_lens[] = {4, 16, 64,128,256};
    const double sparsities[] = {0.5, 0.75, 0.875, 0.9375};

    // ---------------------
    // Up-projection
    // ---------------------
    std::cout << "Start Up-projection benchmark" << std::endl;
    const int up_K[] = {2048, 3072, 4096};
    const int up_N[] = {8192, 8192, 14336};
    const double fixed_sparsity = 0.5;

    // LLaMA-3-1B with different sparsity
      std::cout << "Up-projection with different sparsity" << std::endl;
      for (double s : sparsities) {
          run_test(coyote_thread, 128, up_N[0], up_K[0], s);
      }

    //LLaMA-3-1B with fixed sparsity and changing seq_lens
     std::cout << "Up-projection with increasing seq_len" << std::endl;
     for(int i = 0 ; i < 5; ++i){
         run_test(coyote_thread, seq_lens[i],up_N[0], up_K[0], fixed_sparsity);
     }

    // Changing Model sizes with fixed sparsity and fixed seq_len
      std::cout << "Up-projection with different model sizes" << std::endl;
      for(int i = 0 ; i < 3; ++i){
         run_test(coyote_thread, 64 ,up_N[i], up_K[i], fixed_sparsity);
     }

     std::cout << "End Up-projection benchmark" << std::endl;

    // // ---------------------
    // // Down-projection
    // // ---------------------
    std::cout << "Start Down-projection benchmark" << std::endl;
    const int down_K[] = {8192, 8192, 14336};
    const int down_N[] = {2048, 3072, 4096};

    // LLaMA-3-1B with different sparsity
    std::cout << "Down-projection with different sparsity" << std::endl;
    for (double s : sparsities) {
         run_test(coyote_thread, 128, down_N[0], down_K[0], s);
     }

     // LLaMA-3-1B with different seq_len and fixed sparsity
     std::cout << "Down-projection with increasing seq_len" << std::endl;
      for(int i = 0 ; i < 5; ++i){
         run_test(coyote_thread, seq_lens[i],down_N[0], down_K[0], fixed_sparsity);
     }

      // Changing Model sizes with fixed sparsity and fixed seq_len
      std::cout << "Down-projection with different model sizes" << std::endl;
    for (int i = 0; i < 3; ++i) {
         run_test(coyote_thread, 64, down_N[i], down_K[i], fixed_sparsity);
     }
     std::cout << "End Down-projection benchmark" << std::endl;

     std::cout << "End benchmark" << std::endl;
}

int main(int argc, char *argv[])
{
    // int seq_len = 4;            // M
    // int hidden_size = 1024;     // K
    // int intermediate_size = 32; // N
    // double sparsity = 0.5;

    // Create a Coyote thread and allocate memory for the vectors
    std::unique_ptr<cThread<std::any>> coyote_thread(new cThread<std::any>(DEFAULT_VFPGA_ID, getpid(), 0));
    std::cout << "Coyote Thread created" << std::endl;

     run_benchmark(coyote_thread.get());
    //run_test(coyote_thread.get(), seq_len, intermediate_size, hidden_size, sparsity);

    return EXIT_SUCCESS;
}