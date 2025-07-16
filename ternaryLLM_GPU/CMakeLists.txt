cmake_minimum_required(VERSION 3.20)
project(ter_llm LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)    # Enable CUDA separable compilation globally

message(STATUS "Use Generator: " ${CMAKE_GENERATOR})

# include dir
include_directories(${CMAKE_SOURCE_DIR}/include)

# ter spmm kernels
message(STATUS "Add CUDA SpMM Kernel Code")
add_library(ter_spmm_kernels_obj OBJECT ${CMAKE_SOURCE_DIR}/lib/ter_spmm_kernels.cu)
set_target_properties(ter_spmm_kernels_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ter spmm wrapper
message(STATUS "Add SpMM Wrapper Code")
add_library(ter_spmm_wrapper_obj OBJECT ${CMAKE_SOURCE_DIR}/lib/ter_spmm.cu)
set_target_properties(ter_spmm_wrapper_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(ter_spmm_wrapper_obj PRIVATE ter_spmm_kernels_obj cusparse cublas)

# Create a static library from object files
add_library(ter_spmm_lib STATIC
    $<TARGET_OBJECTS:ter_spmm_kernels_obj>
    $<TARGET_OBJECTS:ter_spmm_wrapper_obj>
)

# ter spmv kernels
message(STATUS "Add CUDA SpMV Kernel Code")
add_library(ter_spmv_kernels_obj OBJECT ${CMAKE_SOURCE_DIR}/lib/ter_spmv_kernels.cu)
set_target_properties(ter_spmv_kernels_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ter spmv wrapper
message(STATUS "Add SpMV Wrapper Code")
add_library(ter_spmv_wrapper_obj OBJECT ${CMAKE_SOURCE_DIR}/lib/ter_spmv.cu)
set_target_properties(ter_spmv_wrapper_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(ter_spmv_wrapper_obj PRIVATE ter_spmv_kernels_obj cusparse cublas)

# Create a static library from object files
add_library(ter_spmv_lib STATIC
    $<TARGET_OBJECTS:ter_spmv_kernels_obj>
    $<TARGET_OBJECTS:ter_spmv_wrapper_obj>
)


add_executable(benchmark_spmm ${CMAKE_SOURCE_DIR}/src/benchmark.cu)
target_link_libraries(benchmark_spmm PRIVATE ter_spmm_lib cusparse cublas)
# add_executable(benchmark_spmv ${CMAKE_SOURCE_DIR}/src/benchmark_spmv.cu)
# target_link_libraries(benchmark_spmv PRIVATE ter_spmv_lib cusparse cublas)