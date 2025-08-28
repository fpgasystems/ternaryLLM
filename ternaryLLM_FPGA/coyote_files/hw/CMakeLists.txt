# CMake configuration
# Coyote has its own internal build system, which must be sourced when synthesizing bitstreams
# This can be done using find_package(CoyoteHW REQUIRED)
cmake_minimum_required(VERSION 3.5)
set(CYT_DIR ${CMAKE_SOURCE_DIR}/../../../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)
find_package(CoyoteHW REQUIRED)

project(perf_GEMM)
message("*** Ternary GEMM [Hardware Build] ***")

set(EN_STRM 0)
set(N_REGIONS 1)
set(EN_MEM 1)
set(N_CARD_AXI 1)
set(EN_ACLK 1)

validation_checks_hw()
    
load_apps (
       VFPGA_C0_0 "src"
    )

 create_hw()
