#pragma once

#include "cuda_lib.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

template <typename T> bool isClose(T a, T b, T percentage_diff_tolerance = 1) {
  T denom = std::max(std::abs(a), std::abs(b));
  if(std::abs(a - b) <= static_cast<T>(1e-6)) {
    return true;
  }
  T percent_diff = (std::abs(a - b) / denom) * static_cast<T>(100.0);
  return percent_diff <= percentage_diff_tolerance;
}

class UNetTest {
public:
  void testMaxPool();
  void testConvolution();
  void testUpsample();
  void testBatchNorm();

private:
  void testMaxPoolForward();
  void testMaxPoolBackward();
  void testConvolutionBackward();
  void testUpsampleBackward();
  void testConvolutionForward();
  void testUpsampleForward();
  void testBatchNormForwardInference();
  void testBatchNormForwardTraining();
  void testBatchNormBackwardTraining();
  template <typename T>
  bool checkArrayEquivalence(T *array_1,
                             T *array_2,
                             size_t length,
                             float precision = 1) {
    for(size_t i = 0; i < length; ++i) {
      if(!isClose(array_1[i], array_2[i], precision)) {
        std::cout << i << ": array 1 val: " << array_1[i]
                  << " array 2 val: " << array_2[i] << std::endl;
        return false;
      }
    }

    return true;
  }
  template <typename T>
  inline void initRandomArray(T *cpu_array, size_t length) {
    for(size_t i = 0; i < length; i++) {
      cpu_array[i] = ((T)std::rand()) / (T)RAND_MAX - 0.5;
    }
  }
};

template <typename T>
inline void printArray(T *arr, int H, int W, std::string label = "") {

  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  std::cout << label << std::endl;
  for(int i = 0; i < H; ++i) {
    for(int j = 0; j < W; ++j) {
      if(arr[i * W + j] < 0) {
        std::cout << std::setprecision(2);
      } else {

        std::cout << std::setprecision(3);
      }
      std::cout << arr[i * W + j] << " ";
    }
    std::cout << std::endl;
  }
}
