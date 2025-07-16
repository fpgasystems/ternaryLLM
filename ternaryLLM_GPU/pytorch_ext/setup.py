from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension

import os
cuda_home_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

# 1. Set the environment variable
print(f"Current CUDA_HOME (before setting): {os.getenv('CUDA_HOME')}")
os.environ["CUDA_HOME"] = cuda_home_path
print(f"CUDA_HOME set to: {os.environ['CUDA_HOME']}")


setup(
    name='ter_spmm',
    version='1.0',
    description='The working version of TAB on CUDA',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='ter_spmm',
            sources=['csrc/ter_spmm_wrapper.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-use_fast_math', '-Xptxas', '-O3']
            }
        ),
    ],
    
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })