#include "ter_spmv.cuh"
#include <chrono>

__host__ void Ternary_SpMV::ter_spmv_cusparse_spmv(TerSparseDataWrap<float>& spmv, SpMMStat& stat)
{
    CUDA_CALL_CHECK(cudaSetDevice(0));
    CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */

    auto fn_s = std::chrono::high_resolution_clock::now();
    /* csc device pointers */
    int* dev_w1_row_indices = 0;
    int* dev_w1_col_offset = 0;
    int8_t* dev_w1_values = 0;

    /* allocate device memory */
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_row_indices), spmv.w1_cnt_nnz*sizeof(int)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_col_offset), (spmv.columns+1)*sizeof(int)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_values), spmv.w1_cnt_nnz*sizeof(float)));

    CUDA_CALL_CHECK(cudaMalloc((void**)(&spmv.dev_x), spmv.size_x*sizeof(float)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&spmv.dev_res), spmv.size_res*sizeof(float)));

    stat.fn_mem_use[stat.curr_config][spmv.sparsity] = (spmv.w1_cnt_nnz*sizeof(int) + (spmv.columns+1)*sizeof(int) + spmv.w1_cnt_nnz*sizeof(float) +
                                                                spmv.size_x*sizeof(float) + spmv.size_res*sizeof(float));

    /* copy csc data */
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_row_indices, (void*)spmv.w1_row_indice, spmv.w1_cnt_nnz*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_col_offset, (void*)spmv.w1_col_offset, (spmv.columns+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_values, (void*)spmv.w1_values, spmv.w1_cnt_nnz*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CALL_CHECK(cudaMemcpy((void*)spmv.dev_x, (void*)spmv.host_x.data(), spmv.size_x*sizeof(float), cudaMemcpyHostToDevice));

    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matW1;
    cusparseDnVecDescr_t vecX, vecRes;

    CUSPARSE_CALL_CHECK(cusparseCreate(&handle))

    // Create sparse matrix W1 in CSC format
    CUSPARSE_CALL_CHECK(cusparseCreateCsr(&matW1, 
                        spmv.columns, spmv.inners, spmv.w1_cnt_nnz, 
                        dev_w1_col_offset, dev_w1_row_indices, dev_w1_values, 
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    // Create dense vector X
    CUSPARSE_CALL_CHECK(cusparseCreateDnVec(&vecX, spmv.size_x, spmv.dev_x, CUDA_R_32F))
    // Create dense vector y
    CUSPARSE_CALL_CHECK(cusparseCreateDnVec(&vecRes, spmv.size_res, spmv.dev_res, CUDA_R_32F))
    
    // allocate an external buffer if needed
    void*                dBuffer    = 0;
    size_t               bufferSize = 0;
    float alpha           = 1.0f;
    float beta            = 0.0f;
    CUSPARSE_CALL_CHECK(cusparseSpMV_bufferSize(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matW1, vecX, &beta, vecRes, CUDA_R_32F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CUDA_CALL_CHECK(cudaMalloc(&dBuffer, bufferSize));

    cudaEvent_t start, stop;
    float kn_span = 0;  // ms
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CUSPARSE_CALL_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matW1, vecX, &beta, vecRes, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kn_span, start, stop);
    stat.kn_spans[stat.curr_config][spmv.sparsity].push_back(kn_span);

    // destroy matrix/vector descriptors
    CUSPARSE_CALL_CHECK(cusparseDestroySpMat(matW1));
    CUSPARSE_CALL_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CALL_CHECK(cusparseDestroyDnVec(vecRes));
    CUSPARSE_CALL_CHECK(cusparseDestroy(handle));

    CUDA_CALL_CHECK(cudaMemcpy((void*)spmv.host_res.data(), (void*)spmv.dev_res, spmv.size_res*sizeof(float), cudaMemcpyDeviceToHost));

    /* free */
    CUDA_CALL_CHECK(cudaFree(dev_w1_col_offset));
    CUDA_CALL_CHECK(cudaFree(dev_w1_row_indices));
    CUDA_CALL_CHECK(cudaFree(dev_w1_values));
    CUDA_CALL_CHECK(cudaFree(dBuffer));
    CUDA_CALL_CHECK(cudaFree(spmv.dev_x));
    CUDA_CALL_CHECK(cudaFree(spmv.dev_w1));
    CUDA_CALL_CHECK(cudaFree(spmv.dev_res));

    auto fn_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fn_span = fn_e - fn_s;
    stat.fn_spans[stat.curr_config][spmv.sparsity].push_back(fn_span.count());

    INFO(spmv.sparsity << " Memory: " << stat.fn_mem_use[stat.curr_config][spmv.sparsity]/1024 << "KB" << " Kernel time: " << kn_span << "ms" << " Function time: " << fn_span.count() << "ms");
}