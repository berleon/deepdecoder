### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_52,code=sm_52)

# Specify include directories
include_directories(
)

# Specify library paths
link_directories(
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_library(
    loss
    mask_split.cu
)
