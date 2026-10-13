#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){

  //here calculate the block grid vector
  const uint block_x = blockIdx.x;
  const uint block_y = blockIdx.y;

  //then transfer the data from global memory to shared memory which is much faster
  __shared__ float shared_A[BLOCKSIZE * BLOCKSIZE];
  __shared__ float shared_B[BLOCKSIZE * BLOCKSIZE];

  //calculate the starting index of a block(advance the pointer to the starting positions)
  A += block_x * BLOCKSIZE * K;                      // row=block_x, col=0
  B += block_y * BLOCKSIZE;                          // row=0, col=block_y
  C += block_x * BLOCKSIZE * N + block_y * BLOCKSIZE;// row=block_x, col=block_y

  //calculate the thread's row and column within the block
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  //initialize the result to zero
  float tmp = 0.0;
  for(int i = 0; i < K; i += BLOCKSIZE) {
    //load the data from global memory to shared memory
    shared_A[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    shared_B[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    //wait for all threads to finish loading data into shared memory
    __syncthreads();

    A += BLOCKSIZE; //advance the pointer to the next block of A
    B += BLOCKSIZE * N; //advance the pointer to the next block of B

    //perform the multiplication and accumulation
    for (int j = 0; j < BLOCKSIZE; ++j) {
        tmp += shared_A[threadRow * BLOCKSIZE + j] * shared_B[j * BLOCKSIZE + threadCol];
    }

    //wait for all threads to finish computing before loading new data into shared memory
    __syncthreads();
  }

  //write the result back to global memory
  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}