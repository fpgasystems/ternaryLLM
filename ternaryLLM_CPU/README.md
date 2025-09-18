# CPU Implementation of Ternary GEMM in Merged TCSC and Uniform TCSC


File Organization
```
GEMM_CPU_FP32.cpp: The source code for FP32 implementation
GEMM_CPU_INT8.cpp: The INT8 version.
initData.hpp:      Init the X and Weights
LlamaModel.cpp:    A simple Llama Model implementation for benchmarking
main.cpp:          The main benchmarking code
TCSC.hpp:          Convert ternary matrix into TCSC formats
SIMD_Generator.ipynb: Generate the SIMD code based on given configurations 
```

This implementation has been tested on the following laptops:
- OS: Windows 11, CPU: AMD Ryzen 7 8845HS (supporting AVX2 and AVX-512), MSVC 2022
- OS: Windows 10, CPU: Intel Core i9-11900H (supporting AVX2 and AVX-512), MSVC 2022

Please include the following folders on your computer. Note that the conda and python virtual environment also have the same folder structure as miniconda3/, so including any of them once is enough. 

Include PyTorch and Eigen library for correct compiling
- libtorch\include OR miniconda3\Lib\site-packages\torch\include
- libtorch\include\torch\csrc\api\include OR miniconda3\Lib\site-packages\torch\include\torch\csrc\api\include
- eigen-3.4.0

Include PyTorch .lib and .dll for correct linking
- libtorch\lib OR miniconda3\Lib\site-packages\torch\lib
- Copy torch.dll and other .dll from libtorch\lib to the .exe folder: x64/release and x64/debug

Setup flags
- Compiler: /Ox /Ot /Oi /AVX2 (/AVX-512) /openmp /fp:fast /GL
- Linker: /LTCG /OPT:REF /OPT:ICF
- Additional compiler flags: /openmp:experimental 
- C/C++ Preprocessor: _CRT_SECURE_NO_WARNINGS (otherwise Eigen and PyTorch won't compile)
