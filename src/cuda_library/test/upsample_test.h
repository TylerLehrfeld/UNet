#pragma once

#include "../cuda_lib.h"
#include "test.h"
#include <cstdlib>

inline void cpuUpsample(const float *input,
                        float *output,
                        const float *weights,
                        int batch_size,
                        int in_channels,
                        int out_channels,
                        int scale,
                        int in_size) {
  int weight_offset = in_channels * out_channels * scale * scale;
  int out_size = in_size * scale;
  for(int b = 0; b < batch_size; ++b) {
    for(int c_out = 0; c_out < out_channels; ++c_out) {
      for(int h = 0; h < out_size; ++h) {
        for(int w = 0; w < out_size; ++w) {
          int out_ind =
              flattenIndex(b, c_out, out_channels, h, out_size, w, out_size);
          // add bias (one per channel)
          output[out_ind] = weights[weight_offset + c_out];
          for(int c_in = 0; c_in < in_channels; ++c_in) {
            int in_ind = flattenIndex(
                b, c_in, in_channels, h / scale, in_size, w / scale, in_size);
            int weight_ind = flattenIndex(
                c_in, c_out, out_channels, h % scale, scale, w % scale, scale);
            output[out_ind] += input[in_ind] * weights[weight_ind];
          }
        }
      }
    }
  }
}

inline void cpuUpsampleBackward(const float *input,
                                const float *output,
                                const float *weights,
                                const float *grad_activations,
                                float *grad_inputs,
                                float *grad_weights,
                                int batch_size,
                                int out_channels,
                                int in_channels,
                                int in_size,
                                int scale) {
  int out_size = scale * in_size;
  int weight_offset = out_channels * in_channels * scale * scale;
  for(int b = 0; b < batch_size; ++b) {
    for(int c_out = 0; c_out < out_channels; ++c_out) {
      for(int h = 0; h < out_size; ++h) {
        for(int w = 0; w < out_size; ++w) {
          int out_ind =
              flattenIndex(b, c_out, out_channels, h, out_size, w, out_size);
          grad_weights[weight_offset + c_out] += grad_activations[out_ind];
          for(int c_in = 0; c_in < in_channels; ++c_in) {
            int in_ind = flattenIndex(
                b, c_in, in_channels, h / scale, in_size, w / scale, in_size);
            int weight_ind = flattenIndex(
                c_in, c_out, out_channels, h % scale, scale, w % scale, scale);
            grad_inputs[in_ind] +=
                grad_activations[out_ind] * weights[weight_ind];
            grad_weights[weight_ind] +=
                grad_activations[out_ind] * input[in_ind];
          }
        }
      }
    }
  }
}

inline void UNetTest::testUpsampleForward() {
  int batch_size = 16;
  int in_channels = 64;
  int out_channels = 32;
  int in_size = 128;
  int scale = 2;

  int out_size = in_size * scale;
  int input_size = batch_size * in_channels * in_size * in_size;
  int output_size = batch_size * out_channels * out_size * out_size;
  int weight_offset = in_channels * out_channels * scale * scale;
  int weight_size = weight_offset + out_channels;
  float *cpu_input = new float[input_size];
  float *cpu_output = new float[output_size];
  float *cpu_weights = new float[weight_size];
  initRandomArray(cpu_input, input_size);
  initRandomArray(cpu_weights, weight_size);
  cpuUpsample(cpu_input,
              cpu_output,
              cpu_weights,
              batch_size,
              in_channels,
              out_channels,
              scale,
              in_size);
  float *gpu_input = cudaGetPointer<float>(input_size);
  float *gpu_output = cudaGetPointer<float>(output_size);

  float *gpu_weights = cudaGetPointer<float>(weight_size);
  cudaLibCopyToDevice(cpu_input, gpu_input, input_size);
  cudaLibCopyToDevice(cpu_weights, gpu_weights, weight_size);
  cudaUpsample(gpu_input,
               gpu_weights,
               gpu_output,
               scale,
               in_channels,
               out_channels,
               in_size,
               in_size,
               batch_size);
  float *gpu_output_host = cudaToCPU(gpu_output, output_size);
  assert(checkArrayEquivalence(gpu_output_host, cpu_output, output_size));
}

inline void UNetTest::testUpsampleBackward() {
  int batch_size = 16;
  int in_channels = 64;
  int out_channels = 32;
  int in_size = 128;
  int scale = 2;

  int out_size = in_size * scale;
  int input_size = batch_size * in_channels * in_size * in_size;
  int output_size = batch_size * out_channels * out_size * out_size;
  int weight_offset = in_channels * out_channels * scale * scale;
  int weight_size = weight_offset + out_channels;
  float *cpu_input = new float[input_size];
  float *cpu_weights = new float[weight_size];

  float *cpu_grad_output = new float[output_size];
  initRandomArray(cpu_input, input_size);
  initRandomArray(cpu_weights, weight_size);
  initRandomArray(cpu_grad_output, output_size);
  float *gpu_input = cudaGetPointer<float>(input_size);
  float *gpu_output = cudaGetPointer<float>(output_size);

  float *gpu_weights = cudaGetPointer<float>(weight_size);
  float *gpu_grad_output = cudaGetPointer<float>(output_size);
  cudaLibCopyToDevice(cpu_input, gpu_input, input_size);
  cudaLibCopyToDevice(cpu_weights, gpu_weights, weight_size);
  cudaLibCopyToDevice(cpu_grad_output, gpu_grad_output, output_size);
  cudaUpsample(gpu_input,
               gpu_weights,
               gpu_output,
               scale,
               in_channels,
               out_channels,
               in_size,
               in_size,
               batch_size);

  float *gpu_grad_input = cudaGetPointer<float>(input_size);
  float *gpu_grad_weights = cudaGetPointer<float>(weight_size);
  float *cpu_grad_input = new float[input_size]();
  float *cpu_grad_weights = new float[weight_size]();
  float *cpu_output = cudaToCPU(gpu_output, output_size);
  cpuUpsampleBackward(cpu_input,
                      cpu_output,
                      cpu_weights,
                      cpu_grad_output,
                      cpu_grad_input,
                      cpu_grad_weights,
                      batch_size,
                      out_channels,
                      in_channels,
                      in_size,
                      scale);
  cudaUpsampleBackward(gpu_grad_output,
                       gpu_input,
                       gpu_weights,
                       gpu_grad_input,
                       gpu_grad_weights,
                       scale,
                       in_channels,
                       out_channels,
                       in_size,
                       in_size,
                       batch_size);
  float *gpu_grad_input_host = cudaToCPU(gpu_grad_input, input_size);
  float *gpu_grad_weights_host = cudaToCPU(gpu_grad_weights, weight_size);

  assert(
      checkArrayEquivalence(gpu_grad_input_host, cpu_grad_input, input_size));
  assert(checkArrayEquivalence(
      gpu_grad_weights_host, cpu_grad_weights, weight_size));
  delete[] gpu_grad_input_host;
  delete[] gpu_grad_weights_host;
  delete[] cpu_output;
  delete[] cpu_grad_weights;
  delete[] cpu_grad_input;
  delete[] cpu_grad_output;
  delete[] cpu_weights;
  delete[] cpu_input;
  cudaLibFree(gpu_input);
  cudaLibFree(gpu_output);
  cudaLibFree(gpu_weights);
  cudaLibFree(gpu_grad_output);
  cudaLibFree(gpu_grad_input);
  cudaLibFree(gpu_grad_weights);
}

inline void UNetTest::testUpsample() {
  testUpsampleForward();
  testUpsampleBackward();

  std::cout << "\033[32mUpsample tests pass!\033[0m" << std::endl;
}
