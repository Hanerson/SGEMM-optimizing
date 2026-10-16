#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm2DBlocktiling(
  int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C
){
  //as always
  //first we need to locate the block

  uint blockRow = blockIdx.y;
  uint blockCol = blockIdx.x;

  //the total elements count divide the elements that a single thread can hold
  const uint totalThreadCountPerBlock = (BM * BN) / (TM * TN);

  assert(blockDim.x == totalThreadCountPerBlock);

  //then calculate the specific reponsible area for a thread in C
  
  uint threadRow = threadIdx.x / (BN / TN);
  uint threadCol = threadIdx.x % (BN / TN);

  //now we know the coordinate in C, naturally we want to push the pointer to the correct place
  A += blockRow * K * BM;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  //ok, we get the correct coordination
  //then we start to move the data into shared MEM
  //but first we need to allocate a perfect space

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  //now we can start the possess of data copying
  //but how to move correct element to correct place?
  //we choose that a single thread move one element per circle
  //we have a smaller number of thread than the target elements required in As and Bs

  uint innerRowA = threadIdx.x / BK;
  uint innerColA = threadIdx.x % BK;
  //that's obviously incorrect
  //in kernel 4, we play this trick cuz we gained the limitation that threadCount must be equal to BK * BN and BM * BK
  //so we have to do somthing more

  //calculate the number of rows one operation can move
  //then we could use a "for --" to fill up the whole As
  uint capableRowsA = totalThreadCountPerBlock / BK;


  //the same with Bs

  uint innerRowB = threadIdx.x / BN;
  uint innerColB = threadIdx.x % BN;

  uint capableRowsB = totalThreadCountPerBlock / BN;

  //for prepariations we allocate space for our temp-results

  float threadResults[TM * TN] = {0.0};
  float regA[TM] = {0.0};
  float regB[TN] = {0.0};

  //start to fill the block tiles
  for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){
    for(uint groupCount = 0; groupCount < BM; groupCount += capableRowsA){
      As[(innerRowA + groupCount) * BK + innerColA] = A[(innerRowA + groupCount) * K + innerColA];
    }

    for(uint groupCount = 0; groupCount < BK; groupCount += capableRowsB){
      Bs[(innerRowB + groupCount) * BN + innerColB] = B[(innerRowB + groupCount) * N + innerColB];
    }

    __syncthreads();

    //ok, now we can advance the pointer to next block;

    A += BK;
    B += N * BK;

    //ok, we can start core calculations now

    for(uint slice = 0; slice < BK; slice ++){
      //we store a slice into register arrays
      //why regA and regB are register-level arrays?
      //that's the compiler's thicky method to upgrade such arrays

      for(uint i = 0; i < TM; i ++){
        regA[i] = As[(threadRow * TM + i) * BK + slice];
      }
      for(uint i = 0; i < TN; ++i) {
        regB[i] = Bs[slice * BN + threadCol * TN + i];
      }

      //calculate
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regA[resIdxM] * regB[resIdxN];
        }
      }
    }
    __syncthreads();
  }
  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}