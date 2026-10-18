#pragma once

#include "kernels/1-naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_1D_blockTiling.cuh"
#include "kernels/5_2D_blockTiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"