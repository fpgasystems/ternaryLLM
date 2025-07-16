#include "ter_spmm.cuh"

/**
 * wrapper function for cusparse spmm api
 */
__host__ void Ternary_SpMM::ter_spmm_cusparse_spmm(TerSparseDataWrap<float>& spmm, SpMMStat& stat) {
    // CUDA_CALL_CHECK(cudaSetDevice(0));
    // CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */
    auto fn_s = std::chrono::high_resolution_clock::now();
    /* csc device pointers */
    int* dev_w1_row_indices = 0;
    int* dev_w1_col_offset = 0;
    int8_t* dev_w1_values = 0;

    /* allocate device memory */
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_row_indices), spmm.w1_cnt_nnz*sizeof(int)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_col_offset), (spmm.columns+1)*sizeof(int)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_w1_values), spmm.w1_cnt_nnz*sizeof(float)));

    CUDA_CALL_CHECK(cudaMalloc((void**)(&spmm.dev_x), spmm.size_x*sizeof(float)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&spmm.dev_res), spmm.size_res*sizeof(float)));

    stat.fn_mem_use[stat.curr_config][spmm.sparsity] = (
        spmm.w1_cnt_nnz*sizeof(int) + 
        (spmm.columns+1)*sizeof(int) + 
        spmm.w1_cnt_nnz*sizeof(float) +
        spmm.size_x*sizeof(float) + spmm.size_res*sizeof(float)
    );

    /* copy csc data */
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_row_indices, (void*)spmm.w1_row_indice, spmm.w1_cnt_nnz*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_col_offset, (void*)spmm.w1_col_offset, (spmm.columns+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_w1_values, (void*)spmm.w1_values, spmm.w1_cnt_nnz*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CALL_CHECK(cudaMemcpy((void*)spmm.dev_x, (void*)spmm.host_x.data(), spmm.size_x*sizeof(float), cudaMemcpyHostToDevice));

    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matW1;
    cusparseDnMatDescr_t matX, matRes;

    CUSPARSE_CALL_CHECK(cusparseCreate(&handle))

    // Create sparse matrix W1 in CSC format
    CUSPARSE_CALL_CHECK(cusparseCreateCsc(&matW1, spmm.inners, spmm.columns, 
                        spmm.w1_cnt_nnz, dev_w1_col_offset, dev_w1_row_indices, dev_w1_values, 
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CUSPARSE_CALL_CHECK(cusparseCreateDnMat(&matX, spmm.rows, spmm.inners, spmm.rows, spmm.dev_x, CUDA_R_32F, CUSPARSE_ORDER_COL));

    CUSPARSE_CALL_CHECK(cusparseCreateDnMat(&matRes, spmm.columns, spmm.rows, spmm.columns, spmm.dev_res, CUDA_R_32F, CUSPARSE_ORDER_COL));
    
    // buffer
    void*                dBuffer    = 0;
    size_t               bufferSize = 0;
    float alpha           = 1.0f;
    float beta            = 0.0f;
    CUSPARSE_CALL_CHECK(cusparseSpMM_bufferSize(handle,
                            CUSPARSE_OPERATION_TRANSPOSE,
                            CUSPARSE_OPERATION_TRANSPOSE,
                            &alpha, matW1, matX, &beta, matRes, CUDA_R_32F,
                            CUSPARSE_SPMM_CSR_ALG1, &bufferSize));
    CUDA_CALL_CHECK(cudaMalloc(&dBuffer, bufferSize));
    
    // 
    cudaEvent_t start, stop;
    float kn_span = 0;  // ms
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    CUSPARSE_CALL_CHECK(cusparseSpMM(handle,
                CUSPARSE_OPERATION_TRANSPOSE,
                CUSPARSE_OPERATION_TRANSPOSE,
                &alpha, matW1, matX, &beta, matRes, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG1, dBuffer));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kn_span, start, stop);
    stat.kn_spans[stat.curr_config][spmm.sparsity].push_back(kn_span);

    // destroy matrix/vector descriptors
    CUSPARSE_CALL_CHECK(cusparseDestroySpMat(matW1));
    CUSPARSE_CALL_CHECK(cusparseDestroyDnMat(matX));
    CUSPARSE_CALL_CHECK(cusparseDestroyDnMat(matRes));
    CUSPARSE_CALL_CHECK(cusparseDestroy(handle));


    CUDA_CALL_CHECK(cudaMemcpy((void*)spmm.host_res.data(), (void*)spmm.dev_res, spmm.size_res*sizeof(float), cudaMemcpyDeviceToHost));

    /* free */
    CUDA_CALL_CHECK(cudaFree(dev_w1_col_offset));
    CUDA_CALL_CHECK(cudaFree(dev_w1_row_indices));
    CUDA_CALL_CHECK(cudaFree(dev_w1_values));
    CUDA_CALL_CHECK(cudaFree(dBuffer));
    CUDA_CALL_CHECK(cudaFree(spmm.dev_x));
    CUDA_CALL_CHECK(cudaFree(spmm.dev_w1));
    CUDA_CALL_CHECK(cudaFree(spmm.dev_res));

    auto fn_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fn_span = fn_e - fn_s;
    stat.fn_spans[stat.curr_config][spmm.sparsity].push_back(fn_span.count());

    INFO(spmm.sparsity << " Memory use: " << stat.fn_mem_use[stat.curr_config][spmm.sparsity]/1024 << "KB"<< " Kernel span: " << kn_span << "ms" << " Function span: " << fn_span.count() << "ms");
    return;
}

/**
 * wrapper function for cublas gemm api
 */
__host__ void Ternary_SpMM::ter_spmm_cublas_spmm(TerSparseDataWrap<float>& spmm, SpMMStat& stat) {
    // CUDA_CALL_CHECK(cudaSetDevice(0));
    // CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */

    cublasHandle_t cublasH = NULL;
    // cudaStream_t stream = NULL;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    float* dev_w1_float = 0;
    auto fn_s = std::chrono::high_resolution_clock::now();  // function runtime start
    auto prekn_s = std::chrono::high_resolution_clock::now();   // pre-kernel runtime start

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CALL_CHECK(cublasCreate(&cublasH));

    // CUDA_CALL_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUBLAS_CALL_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CALL_CHECK(cudaMalloc(reinterpret_cast<void **>(&spmm.dev_x), sizeof(float) * spmm.size_x));
    CUDA_CALL_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_w1_float), sizeof(float) * spmm.size_w1));
    CUDA_CALL_CHECK(cudaMalloc(reinterpret_cast<void **>(&spmm.dev_res), sizeof(float) * spmm.size_res));
    stat.fn_mem_use[stat.curr_config][spmm.sparsity] = (
        sizeof(float) * spmm.size_x + 
        sizeof(float) * spmm.size_w1 + 
        sizeof(float) * spmm.size_res
    );

    CUDA_CALL_CHECK(cudaMemcpy(spmm.dev_x, spmm.host_x.data(), sizeof(float) * spmm.size_x, cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy(dev_w1_float, spmm.host_w1_xtype.data(), sizeof(float) * spmm.size_w1, cudaMemcpyHostToDevice));

    auto prekn_e = std::chrono::high_resolution_clock::now();   // pre-kernel runtime end
    std::chrono::duration<double, std::milli> prekn_span = prekn_e - prekn_s; // data prepare duration before API call


    /* step 3: compute */
    const float alpha = 1.0;
    const float beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasSgemm(cublasH, transa, transb, spmm.rows, spmm.columns, spmm.inners, &alpha, 
                spmm.dev_x, spmm.rows, 
                dev_w1_float, spmm.inners, &beta, 
                spmm.dev_res, spmm.rows);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kn_span = 0;
    cudaEventElapsedTime(&kn_span, start, stop);
    stat.kn_spans[stat.curr_config][spmm.sparsity].push_back(kn_span);

    auto postkn_s = std::chrono::high_resolution_clock::now();  // post-kernel start

    /* step 4: copy data to host */
    CUDA_CALL_CHECK(cudaMemcpy(spmm.host_res.data(), spmm.dev_res, sizeof(float) * spmm.size_res, cudaMemcpyDeviceToHost));


    /* free resources */
    CUDA_CALL_CHECK(cudaFree(spmm.dev_res));
    CUDA_CALL_CHECK(cudaFree(spmm.dev_x));
    CUDA_CALL_CHECK(cudaFree(dev_w1_float));

    CUBLAS_CALL_CHECK(cublasDestroy(cublasH));

    // CUDA_CALL_CHECK(cudaStreamDestroy(stream));

    // CUDA_CALL_CHECK(cudaDeviceReset());
    
    auto fn_e = std::chrono::high_resolution_clock::now();  // function runtime end
    std::chrono::duration<double, std::milli> postkn_span = fn_e - postkn_s;    // post-kernel duration
    std::chrono::duration<double, std::milli> fn_span = fn_e - fn_s;            // function duration
    stat.fn_spans[stat.curr_config][spmm.sparsity].push_back(fn_span.count());
    
    INFO(spmm.sparsity << " Memory: " << stat.fn_mem_use[stat.curr_config][spmm.sparsity]/1024 << "KB"<< " Kernel: " << kn_span << "ms"<< " Pre: " << prekn_span.count() << "ms"<< " Post: " << postkn_span.count() << "ms"<< " Function: " << fn_span.count() << "ms");
}