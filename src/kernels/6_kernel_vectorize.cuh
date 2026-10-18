#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, float alpha, const float *A. const float *B. float beta, float *C){
  //as always
  //first calculate the block index (coordinate)

  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  //allocate space for the current blocktile in smem
  //attention here: we still make As as BM * BK
  //but actually we use As as BK * BM
  //inorder to enhance cache
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  //then we need to calculate the inner thread coordinate
  //here is the first optimize method 
  //we use type float4 to read 4 nums at a time
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  //the only difference from kernel 5 is one thread take responsible of 4 elements

  //then we do the same with kernel 5
  //allocate thread-local cache for results in registerfile

  float threadResults[TM * TN] = {0.0};
  float regA[TM] = {0.0};
  float regB[TN] = {0.0};

  //block sliding window: maybe this name is more sensible
  for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){
    //TODO: data transfer and core calculate
  }

  //TODO: write back the caiculate date
}