#include "cuda_lib.h"
#include "cuda_runtime.h"
#include <cassert>
#include <iostream>

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

float cuda_add_nums() {
  float A[3] = {0, 1, 2};
  float B[3] = {2, 3, 4};
  float C[3];
  float *dA;
  float *dB;
  float *dC;
  cudaMalloc(&dA, 3 * sizeof(float));
  cudaMalloc(&dB, 3 * sizeof(float));
  cudaMalloc(&dC, 3 * sizeof(float));
  cudaMemcpy(dA, A, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, 3 * sizeof(float), cudaMemcpyHostToDevice);
  vectorAdd<<<1, 3>>>(dA, dB, dC, 3);
  cudaMemcpy(C, dC, 3 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "asserts starting" << std::endl;
  for (int i = 0; i < 3; i++) {
    assert(C[i] == A[i] + B[i]);
  }

  std::cout << "asserts ended" << std::endl;
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return C[0];
}
