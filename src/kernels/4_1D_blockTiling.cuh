#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<const int BM, const int BN, const int BK, const int TM>

__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float *C){
  //first always calculate the block coordinate

  //interestingly here to enhance the L2 cache
  //we use different assignment
  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  const uint threadRow = threadIdx.x / BN;
  const uint threadCol = threadIdx.x % BN;

  //push the pointer to the calculate point
  C += blockRow * BM * N + blockCol * BN;
  A += blockRow * BM * K;
  B += blockCol * BN;

  //currently has not started the calculation
  //we only use every thread to locate the potential data in global memory
  //which will be used in the future calculation in SMEM
  const uint threadACol = threadIdx.x % BK;
  const uint threadARow = threadIdx.x / BK;
  const uint threadBCol = threadIdx.x % BN;
  const uint threadBRow = threadIdx.x / BN;

  //now we have all the needed location
  //just need to load the data from global memory to shared memory

  //first allocate need space in shared memory

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  //we asked for a TM * float space
  //cuz a thread take responsiability to calculate TM elements
  float ans_container[TM] = {0.0};

  for(int i = 0; i < K; i += BK){
    //very important!!! here the pointer of A has been biased previously
    //so you can mentally consider A and As start the same place
    //only difference is the width
    //which As is BK, A is K
    As[threadARow * BK + threadACol] = A[threadARow * K + threadACol];
    Bs[threadBRow * BN + threadBCol] = B[threadBRow * N + threadBCol];

    __syncthreads();

    //now A and B has compeleted partial work, we can advance the two to the next place
    A += BK;
    B += BK * N;

    //congratulations!
    //we can start core calculations now!
    //first declear a temp arr to store our mid-results

    for(uint verticalCount = 0; verticalCount < BK; verticalCount ++){
      float targetBElement = Bs[verticalCount * BN + threadCol];

      for(uint i = 0; i < TM; i ++){
        //verticalCount is the index of the current column in A and the current row in B
        ans_container[i] += targetBElement * As[(threadRow * TM + i) * BK + verticalCount];
      }
    }

    __syncthreads();
  }

  //now we have completed the calculation of the current block
  //we need to write the result back to global memory
  for(uint i = 0; i < TM; i ++){
    C[(threadRow * TM + i) * N + threadCol] = ans_container[i] * alpha + beta * C[(threadRow * TM + i) * N + threadCol];
  }
}