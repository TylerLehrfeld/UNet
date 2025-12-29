#pragma once

#include "../cuda_lib.h"
#include "test.h"
#include <cstdlib>

// Do a 2D convolution on the CPU.
// input: flattened input image: I[batch][channel][H][W]
// weights: flattened kernels W[in channel][out channel][H][W].
// Biases are at W[in channels * out channels * K * K + out channel]
// activations: flattened output image: O[batch][channel][H][W]
template <typename T>
inline void cpuConvolve(T *input,
                        T *weights,
                        T *activations,
                        int in_height,
                        int in_width,
                        int padding,
                        int stride,
                        int in_channels,
                        int out_channels,
                        int kernel_size,
                        int batch_size) {
  int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
  int center_h = kernel_size / 2;
  int center_w = kernel_size / 2;
  for(int b = 0; b < batch_size; ++b) {

    for(int c_out = 0; c_out < out_channels; ++c_out) {
      for(int h = 0; h < out_height; ++h) {
        for(int w = 0; w < out_width; ++w) {
          int in_h = h * stride - padding + kernel_size / 2;
          int in_w = w * stride - padding + kernel_size / 2;
          int out_ind =
              flattenIndex(b, c_out, out_channels, h, out_height, w, out_width);
          activations[out_ind] =
              weights[in_channels * out_channels * kernel_size * kernel_size +
                      c_out];
          for(int c_in = 0; c_in < in_channels; ++c_in) {

            for(int kh = 0; kh < kernel_size; ++kh) {
              for(int kw = 0; kw < kernel_size; ++kw) {
                int cur_h = in_h - center_h + kh;
                int cur_w = in_w - center_w + kw;

                if(cur_h >= 0 && cur_h < in_height && cur_w >= 0 &&
                   cur_w < in_width) {
                  int in_ind = flattenIndex(
                      b, c_in, in_channels, cur_h, in_height, cur_w, in_width);
                  int weights_ind = flattenIndex(c_in,
                                                 c_out,
                                                 out_channels,
                                                 kh,
                                                 kernel_size,
                                                 kw,
                                                 kernel_size);
                  activations[out_ind] += input[in_ind] * weights[weights_ind];
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
inline void cpuConvolutionBackwardWeightsAndInputs(const T *input,
                                                   const T *activations,
                                                   const T *grad_activations,
                                                   const T *weights,
                                                   T *weight_grads,
                                                   T *input_grads,
                                                   int in_size,
                                                   int in_channels,
                                                   int out_channels,
                                                   int padding,
                                                   int stride,
                                                   int kernel_size,
                                                   int batch_size,
                                                   int out_size) {
  for(int batch = 0; batch < batch_size; batch++) {
    for(int c_out = 0; c_out < out_channels; c_out++) {
      for(int c_in = 0; c_in < in_channels; c_in++) {
        int start = -padding + kernel_size / 2;
        int end = in_size + padding - kernel_size / 2;
        int out_h = 0;
        for(int h = start; h < end; h += stride) {
          int out_w = 0;
          for(int w = start; w < end; w += stride) {
            for(int kh = 0; kh < kernel_size; kh++) {
              for(int kw = 0; kw < kernel_size; kw++) {
                int centered_kh = kh - kernel_size / 2;
                int centered_kw = kw - kernel_size / 2;
                if(h + centered_kh >= 0 && h + centered_kh < in_size &&
                   w + centered_kw < in_size && w + centered_kw >= 0) {
                  int weights_ind = flattenIndex(c_in,
                                                 c_out,
                                                 out_channels,
                                                 kh,
                                                 kernel_size,
                                                 kw,
                                                 kernel_size);

                  int input_ind = flattenIndex(batch,
                                               c_in,
                                               in_channels,
                                               h + centered_kh,
                                               in_size,
                                               w + centered_kw,
                                               in_size);
                  int output_ind = flattenIndex(batch,
                                                c_out,
                                                out_channels,
                                                out_h,
                                                out_size,
                                                out_w,
                                                out_size);
                  input_grads[input_ind] +=
                      weights[weights_ind] * grad_activations[output_ind];
                  weight_grads[weights_ind] +=
                      grad_activations[output_ind] * input[input_ind];
                }
              }
            }
            out_w++;
          }
          out_h++;
        }
      }
    }
  }
}
template <typename T>
inline void cpuConvolutionBackwardBiases(const T *weights,
                                         const T *grad_activations,
                                         T *grad_weights,
                                         int weights_offset,
                                         int size,
                                         int batch_size,
                                         int in_channels,
                                         int out_channels) {
  for(int b = 0; b < batch_size; b++) {
    for(int c = 0; c < out_channels; c++) {
      for(int h = 0; h < size; h++) {
        for(int w = 0; w < size; ++w) {
          int out_ind = flattenIndex(b, c, out_channels, h, size, w, size);
          grad_weights[weights_offset + c] += grad_activations[out_ind];
        }
      }
    }
  }
}

inline void UNetTest::testConvolutionBackward() {
  int in_size = 256;
  int kernel_size = 3;
  int padding = 1;
  int stride = 1;
  int batch_size = 16;
  int in_channels = 3;
  int out_channels = 64;
  int out_size = (in_size + 2 * padding - kernel_size) / stride + 1;

  int kernel_size_squared = kernel_size * kernel_size;
  int input_size = in_size * in_size * batch_size * in_channels;
  int output_size = out_size * out_size * batch_size * out_channels;

  int weights_offset = in_channels * out_channels * kernel_size_squared;
  int weight_size = weights_offset + out_channels;
  float *cpu_input = new float[input_size];
  float *cpu_kernels_and_biases = new float[weight_size]();
  float *cpu_grad_kernels_and_biases = new float[weight_size]();
  float *cpu_activations = new float[output_size];
  float *cpu_dLdY = new float[output_size]();
  float *cpu_dYdX = new float[output_size]();
  initRandomArray(cpu_dLdY, output_size);
  initRandomArray(cpu_input, input_size);
  initRandomArray(cpu_kernels_and_biases, weight_size);
  // printArray(cpu_input, in_size, in_size, "cpu input");
  // printArray(cpu_dLdY, out_size, out_size, "cpu output gradients");
  float *gpu_input = cudaGetPointer<float>(input_size);
  cudaLibCopyToDevice(cpu_input, gpu_input, input_size);
  float *gpu_weights = cudaGetPointer<float>(weight_size);
  cudaLibCopyToDevice(cpu_kernels_and_biases, gpu_weights, weight_size);
  float *gpu_activations = cudaGetPointer<float>(output_size);

  cudaConvolve(gpu_weights,
               gpu_input,
               gpu_activations,
               output_size,
               batch_size,
               kernel_size,
               in_channels,
               out_channels,
               out_size,
               out_size,
               in_size,
               in_size,
               padding,
               stride);
  float *gpu_dLdY = cudaGetPointer<float>(output_size);
  cudaLibCopyToDevice(cpu_dLdY, gpu_dLdY, output_size);
  float *gpu_dYdX = cudaGetPointer<float>(input_size);

  cpuConvolutionBackwardBiases(cpu_kernels_and_biases,
                               cpu_dLdY,
                               cpu_grad_kernels_and_biases,
                               weights_offset,
                               out_size,
                               batch_size,
                               in_channels,
                               out_channels);
  cpuConvolutionBackwardWeightsAndInputs(cpu_input,
                                         cpu_activations,
                                         cpu_dLdY,
                                         cpu_kernels_and_biases,
                                         cpu_grad_kernels_and_biases,
                                         cpu_dYdX,
                                         in_size,
                                         in_channels,
                                         out_channels,
                                         padding,
                                         stride,
                                         kernel_size,
                                         batch_size,
                                         out_size);

  // printArray(cpu_dYdX, in_size, in_size, "cpu gradient inputs");
  // printArray(cpu_grad_kernels_and_biases,
  //        kernel_size,
  //       kernel_size,
  //      "kernel gradients");
  float *gpu_grad_weights = cudaGetPointer<float>(weight_size);
  cudaConvolveBackward(gpu_dLdY,
                       gpu_input,
                       gpu_weights,
                       gpu_dYdX,
                       gpu_grad_weights,
                       batch_size,
                       kernel_size,
                       in_channels,
                       out_channels,
                       out_size,
                       out_size,
                       in_size,
                       in_size,
                       padding,
                       stride);
  float *gpu_dYdX_host = cudaToCPU(gpu_dYdX, input_size);
  float *gpu_grad_weights_host = cudaToCPU(gpu_grad_weights, weight_size);
  // printArray(
  //   gpu_grad_weights_host, kernel_size, kernel_size, "gpu kernel
  //   gradients");
  // printArray(gpu_dYdX_host, in_size, in_size, "gpu dYdX");
  assert(checkArrayEquivalence(cpu_dYdX, gpu_dYdX_host, input_size));
  assert(checkArrayEquivalence(
      cpu_grad_kernels_and_biases, gpu_grad_weights_host, weight_size));
}
inline void UNetTest::testConvolutionForward() {
  int in_size = 256;
  int kernel_size = 3;
  int padding = 1;
  int stride = 1;
  int batch_size = 16;
  int in_channels = 3;
  int out_channels = 64;
  int out_size = (in_size + 2 * padding - kernel_size) / stride + 1;
  int input_size = in_size * in_size * batch_size * in_channels;
  int output_size = out_size * out_size * batch_size * out_channels;
  int weight_size =
      in_channels * out_channels * kernel_size * kernel_size + out_channels;
  float *cpu_input = new float[input_size];
  float *cpu_kernels_and_biases = new float[weight_size];
  float *cpu_activations = new float[output_size];
  initRandomArray(cpu_input, input_size);
  initRandomArray(cpu_kernels_and_biases, weight_size);

  float *gpu_input = cudaGetPointer<float>(input_size);
  cudaLibCopyToDevice(cpu_input, gpu_input, input_size);
  float *gpu_weights = cudaGetPointer<float>(weight_size);
  cudaLibCopyToDevice(cpu_kernels_and_biases, gpu_weights, weight_size);
  float *gpu_activations = cudaGetPointer<float>(output_size);
  cpuConvolve<float>(cpu_input,
                     cpu_kernels_and_biases,
                     cpu_activations,
                     in_size,
                     in_size,
                     padding,
                     stride,
                     in_channels,
                     out_channels,
                     kernel_size,
                     batch_size);
  cudaConvolve(gpu_weights,
               gpu_input,
               gpu_activations,
               output_size,
               batch_size,
               kernel_size,
               in_channels,
               out_channels,
               out_size,
               out_size,
               in_size,
               in_size,
               padding,
               stride);
  float *gpu_output_host = cudaToCPU(gpu_activations, output_size);
  assert(checkArrayEquivalence(gpu_output_host, cpu_activations, output_size));
  delete[] cpu_activations;
  delete[] gpu_output_host;
  delete[] cpu_input;
  delete[] cpu_kernels_and_biases;
  cudaLibFree(gpu_input);
  cudaLibFree(gpu_activations);
  cudaLibFree(gpu_weights);
}

inline void UNetTest::testConvolution() {
  testConvolutionBackward();
  testConvolutionForward();
  std::cout << "\033[32mConvolution tests pass!\033[0m" << std::endl;
}
