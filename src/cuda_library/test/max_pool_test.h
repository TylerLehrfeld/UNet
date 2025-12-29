#pragma once

#include "../cuda_lib.h"
#include "test.h"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>

inline void cpuForwardMaxPool(float *input_arr,
                              float *output_arr,
                              int batch_size,
                              int C,
                              int in_H,
                              int in_W,
                              int stride) {
  for(int batch = 0; batch < batch_size; batch++) {
    for(int channel = 0; channel < C; channel++) {
      for(int h = 0; h < in_H / stride; h++) {
        for(int w = 0; w < in_W / stride; w++) {
          int start_ind = flattenIndex(
              batch, channel, C, h * stride, in_H, w * stride, in_W);
          float max = input_arr[start_ind];
          for(int kh = 0; kh < stride; kh++) {
            for(int kw = 0; kw < stride; kw++) {
              int input_index = start_ind + kh * in_H + kw;
              float in_val = input_arr[input_index];
              if(in_val > max) {
                max = in_val;
              }
            }
          }
          output_arr[flattenIndex(
              batch, channel, C, h, in_H / stride, w, in_W / stride)] = max;
        }
      }
    }
  }
}

inline void UNetTest::testMaxPoolForward() {
  // Define test dimensions
  int in_height = 256;
  int in_width = 256;
  int in_channels = 64;
  int batch_size = 8;
  int stride = 2;

  // calculate sizes for arrays
  int total_size = in_height * in_width * in_channels * batch_size;
  int output_size = total_size / (stride * stride);

  // allocate arrays
  float *cpu_input = new float[total_size];
  float *cpu_output = new float[output_size];
  float *d_input_array = cudaGetPointer<float>(total_size);
  float *d_output_array = cudaGetPointer<float>(output_size);

  float *gpu_output;

  // init cpu array, copy cpu to gpu input
  initRandomArray(cpu_input, total_size);
  cudaLibCopyToDevice(cpu_input, d_input_array, total_size);

  // verify copy
  float *val = cudaToCPU(d_input_array, 1);
  assert(cpu_input[0] == *val);
  delete[] val;

  // Do operations
  cudaMaxPool(d_input_array,
              d_output_array,
              batch_size,
              stride,
              stride,
              in_channels,
              in_height,
              in_width);

  cpuForwardMaxPool(cpu_input,
                    cpu_output,
                    batch_size,
                    in_channels,
                    in_height,
                    in_width,
                    stride);

  // do operations
  // verify output
  val = cudaToCPU(d_output_array, 1);
  assert(isClose(cpu_output[0], *val));
  delete[] val;

  gpu_output = cudaToCPU(d_output_array, output_size);
  assert(checkArrayEquivalence(cpu_output, gpu_output, output_size));

  cudaLibFree(d_input_array);
  cudaLibFree(d_output_array);
  delete[] gpu_output;
  delete[] cpu_output;
}

inline void cpuMaxPoolBackwards(const float *grad_activations,
                                const float *inputs,
                                const float *activations,
                                float *grad_inputs,
                                int batch_size,
                                int stride,
                                int in_channels,
                                int in_height,
                                int in_width) {
  int out_height = in_height / stride;
  int out_width = in_width / stride;
  for(int h = 0; h < in_height; ++h) {
    for(int w = 0; w < in_width; ++w) {
      for(int c = 0; c < in_channels; ++c) {
        for(int b = 0; b < batch_size; ++b) {
          int input_ind =
              flattenIndex(b, c, in_channels, h, in_height, w, in_width);
          int out_h = h / stride;
          int out_w = w / stride;

          int output_ind = flattenIndex(
              b, c, in_channels, out_h, out_height, out_w, out_width);
          if(inputs[input_ind] == activations[output_ind]) {
            grad_inputs[input_ind] = grad_activations[output_ind];
          } else {
            grad_inputs[input_ind] = 0;
          }
        }
      }
    }
  }
}

inline void UNetTest::testMaxPoolBackward() {
  int in_height = 128;
  int in_width = 128;
  int in_channels = 64;
  int batch_size = 16;
  int stride = 2;
  // calculate sizes for arrays
  int total_size = in_height * in_width * in_channels * batch_size;
  int output_size = total_size / (stride * stride);

  // allocate arrays
  float *cpu_input = new float[total_size];
  float *cpu_output = new float[output_size];
  float *cpu_dYdL = new float[output_size];
  float *cpu_dXdY = new float[total_size];

  float *gpu_input = cudaGetPointer<float>(total_size);
  float *gpu_output = cudaGetPointer<float>(output_size);
  float *gpu_dYdL = cudaGetPointer<float>(output_size);
  float *gpu_dXdY = cudaGetPointer<float>(total_size);

  // init cpu array, copy cpu to gpu input
  initRandomArray(cpu_input, total_size);
  cudaLibCopyToDevice(cpu_input, gpu_input, total_size);
  cpuForwardMaxPool(cpu_input,
                    cpu_output,
                    batch_size,
                    in_channels,
                    in_height,
                    in_width,
                    stride);

  cudaLibCopyToDevice(cpu_output, gpu_output, output_size);
  // init simulated gradients from next layer
  initRandomArray(cpu_dYdL, output_size);

  cudaLibCopyToDevice(cpu_dYdL, gpu_dYdL, output_size);

  cpuMaxPoolBackwards(cpu_dYdL,
                      cpu_input,
                      cpu_output,
                      cpu_dXdY,
                      batch_size,
                      stride,
                      in_channels,
                      in_height,
                      in_width);

  cudaMaxPoolBackward(gpu_dYdL,
                      gpu_input,
                      gpu_output,
                      gpu_dXdY,
                      batch_size,
                      stride,
                      in_channels,
                      in_height,
                      in_width);

  float *gpu_dXdY_host = cudaToCPU(gpu_dXdY, total_size);
  assert(checkArrayEquivalence(gpu_dXdY_host, cpu_dXdY, total_size));
}

inline void UNetTest::testMaxPool() {
  testMaxPoolForward();
  testMaxPoolBackward();
  std::cout << "\033[32mMax pool tests pass!\033[0m" << std::endl;
}
