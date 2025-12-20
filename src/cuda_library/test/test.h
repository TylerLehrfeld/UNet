#pragma once

#include "cuda_lib.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

template <typename T> bool isClose(T a, T b, T precision = 5) {
  T eps = std::pow(10, -precision);
  return std::abs(a - b) < eps;
}
class UNetTest {
public:
  void testMaxPool();
  void testConvolution();

private:
  void testMaxPoolForward();
  void testMaxPoolBackward();
  void testConvolutionBackward();
  void testConvolutionForward();
  template <typename T>
  bool checkArrayEquivalence(T *array_1, T *array_2, size_t length) {
    for(size_t i = 0; i < length; ++i) {
      if(!isClose(array_1[i], array_2[i])) {

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

inline int
flattenIndex(int batch_num, int c, int C, int h, int H, int w, int W) {
  return batch_num * C * H * W + c * H * W + h * W + w;
}
