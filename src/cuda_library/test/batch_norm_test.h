#include "../cuda_lib.h"
#include "test.h"
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

inline void cpuBatchNormTrain(const float *activations,
                              const float *weights,
                              float *normalized_activations,
                              int channels,
                              int size,
                              int batch_size) {

  int channel_N = size * size * batch_size;
  float eps = 1e-8;
  for(int channel = 0; channel < channels; ++channel) {
    float channel_sum = 0;
    float channel_sum_squared = 0;

    for(int b = 0; b < batch_size; ++b) {
      for(int h = 0; h < size; ++h) {
        for(int w = 0; w < size; ++w) {
          int index = flattenIndex(b, channel, channels, h, size, w, size);
          channel_sum += activations[index];
          channel_sum_squared += activations[index] * activations[index];
        }
      }
    }
    float channel_mean = channel_sum / channel_N;
    float channel_variance =
        channel_sum_squared / channel_N - channel_mean * channel_mean;

    for(int b = 0; b < batch_size; ++b) {
      for(int h = 0; h < size; ++h) {
        for(int w = 0; w < size; ++w) {

          int input_index =
              flattenIndex(b, channel, channels, h, size, w, size);
          normalized_activations[input_index] =
              weights[channel * 2] *
                  ((activations[input_index] - channel_mean) /
                   (sqrt(channel_variance) + eps)) +
              weights[channel * 2 + 1];
        }
      }
    }
  }
}

inline void cpuBatchNormInference(const float *activations,
                                  float *normalized_activations,
                                  const float *weights,
                                  const float *test_stats,
                                  int channels,
                                  int size) {
  float eps = 1e-8f;
  for(int channel = 0; channel < channels; ++channel) {
    float channel_mean = test_stats[channel * 2];
    float channel_std = sqrt(test_stats[channel * 2 + 1] + eps);
    for(int h = 0; h < size; ++h) {
      for(int w = 0; w < size; ++w) {
        int input_index = flattenIndex(0, channel, channels, h, size, w, size);
        normalized_activations[input_index] =
            weights[channel * 2] *
                ((activations[input_index] - channel_mean) / channel_std) +
            weights[channel * 2 + 1];
      }
    }
  }
}

inline void cpuBatchNormBackwards(const float *input,
                                  const float *output_gradients,
                                  const float *weights,
                                  const float *test_stats,
                                  float *input_gradients,
                                  float *weight_gradients,
                                  int channels,
                                  int batch_size,
                                  int size) {
  float norm_size = size * size * batch_size;
  float eps = 1e-8;
  for(int channel = 0; channel < channels; ++channel) {
    // zero out gradients to start
    int gamma_ind = channel * 2;
    int beta_ind = channel * 2 + 1;
    weight_gradients[gamma_ind] = 0;
    weight_gradients[beta_ind] = 0;
    float gamma = weights[gamma_ind];
    float beta = weights[beta_ind];
    float mean = test_stats[gamma_ind] / (norm_size);
    float variance = (test_stats[beta_ind] / (norm_size)) - (mean * mean);
    // if(channel == 0) {
    //   std::cout << "mean: " << mean << " variance: " << variance <<
    //   std::endl;
    // }
    for(int batch = 0; batch < batch_size; ++batch) {
      for(int h = 0; h < size; ++h) {
        for(int w = 0; w < size; ++w) {

          int index = flattenIndex(batch, channel, channels, h, size, w, size);
          float x_hat = (input[index] - mean) / sqrt(variance + eps);
          weight_gradients[gamma_ind] += output_gradients[index] * x_hat;
          weight_gradients[beta_ind] += output_gradients[index];
        }
      }
    }
    float sum_dy = weight_gradients[beta_ind];

    float sum_dy_xhat = weight_gradients[gamma_ind];

    for(int batch = 0; batch < batch_size; ++batch) {
      for(int h = 0; h < size; ++h) {
        for(int w = 0; w < size; ++w) {
          int index = flattenIndex(batch, channel, channels, h, size, w, size);

          float x_hat = (input[index] - mean) / sqrt(variance + eps);
          float accumulated = output_gradients[index] -
                              1.0f / norm_size * (sum_dy + x_hat * sum_dy_xhat);
          input_gradients[index] = gamma / sqrt(variance + eps) * accumulated;
        }
      }
    }
  }
}

inline void UNetTest::testBatchNormForwardInference() {
  int in_height = 256;
  int in_width = in_height;
  int channels = 64;
  int size = in_height;
  int total_size = in_height * in_width * channels;
  float *activations = new float[total_size];
  float *normalized_activations = new float[total_size];
  float *weights = new float[channels * 2];
  float *test_stats = new float[channels * 2];
  initRandomArray(activations, total_size);
  initRandomArray(weights, channels * 2);
  initRandomArray(test_stats, channels * 2);
  for(int i = 1; i < channels * 2; i += 2) {
    test_stats[i] = abs(test_stats[i]);
  }
  cpuBatchNormInference(
      activations, normalized_activations, weights, test_stats, channels, size);
  float *gpu_input = cudaGetPointer<float>(total_size);
  float *gpu_activations = cudaGetPointer<float>(total_size);
  float *gpu_weights = cudaGetPointer<float>(channels * 2);
  float *gpu_test_stats = cudaGetPointer<float>(channels * 2);
  cudaLibCopyToDevice(activations, gpu_input, total_size);
  cudaLibCopyToDevice(weights, gpu_weights, channels * 2);
  cudaLibCopyToDevice(test_stats, gpu_test_stats, channels * 2);
  cudaBatchNormInference(gpu_input,
                         gpu_activations,
                         gpu_weights,
                         gpu_test_stats,
                         size,
                         size,
                         channels);
  float *gpu_normalized_activations_host =
      cudaToCPU(gpu_activations, total_size);
  assert(checkArrayEquivalence(
      gpu_normalized_activations_host, normalized_activations, total_size));
  delete[] activations;
  delete[] normalized_activations;
  delete[] weights;
  delete[] test_stats;
  cudaLibFree(gpu_input);
  cudaLibFree(gpu_activations);
  cudaLibFree(gpu_weights);
  cudaLibFree(gpu_test_stats);
  delete[] gpu_normalized_activations_host;
}

inline void UNetTest::testBatchNormForwardTraining() {
  int in_height = 256;
  int in_width = in_height;
  int channels = 64;
  int batch_size = 16;
  int size = in_height;
  int total_size = in_height * in_width * channels * batch_size;
  float *activations = new float[total_size];
  float *normalized_activations = new float[total_size];
  float *weights = new float[channels * 2];
  float *running_stats = new float[channels * 2];
  initRandomArray(activations, total_size);
  initRandomArray(weights, channels * 2);
  initRandomArray(running_stats, channels * 2);
  for(int i = 1; i < channels * 2; i += 2) {
    running_stats[i] = abs(running_stats[i]);
  }
  cpuBatchNormTrain(
      activations, weights, normalized_activations, channels, size, batch_size);

  float *gpu_input = cudaGetPointer<float>(total_size);
  float *gpu_activations = cudaGetPointer<float>(total_size);
  float *gpu_weights = cudaGetPointer<float>(channels * 2);
  float *gpu_batch_stats = cudaGetPointer<float>(channels * 2);
  float *gpu_running_stats = cudaGetPointer<float>(channels * 2);
  cudaLibCopyToDevice(activations, gpu_input, total_size);
  cudaLibCopyToDevice(weights, gpu_weights, channels * 2);
  cudaLibCopyToDevice(running_stats, gpu_running_stats, channels * 2);
  cudaBatchNormTrain(gpu_input,
                     gpu_activations,
                     gpu_weights,
                     gpu_batch_stats,
                     gpu_running_stats,
                     channels,
                     size,
                     size,
                     batch_size);
  float *gpu_normalized_activations_host =
      cudaToCPU(gpu_activations, total_size);
  // Higher percent tolerance here because of atomic add for accumulating
  // gradients
  assert(checkArrayEquivalence(
      gpu_normalized_activations_host, normalized_activations, total_size, 2));
  delete[] activations;
  delete[] normalized_activations;
  delete[] weights;
  delete[] running_stats;
  cudaLibFree(gpu_input);
  cudaLibFree(gpu_activations);
  cudaLibFree(gpu_weights);
  cudaLibFree(gpu_batch_stats);
  cudaLibFree(gpu_running_stats);
  delete[] gpu_normalized_activations_host;
}

inline void UNetTest::testBatchNormBackwardTraining() {
  int size = 256;
  int batch_size = 16;
  int channels = 64;
  int total_size = size * size * batch_size * channels;
  int param_size = channels * 2;
  float *cpu_input = new float[total_size];
  float *cpu_output_gradients = new float[total_size];
  float *cpu_weights = new float[param_size];
  float *cpu_input_gradients = new float[total_size]();
  float *cpu_weight_gradients = new float[param_size]();

  initRandomArray(cpu_input, total_size);
  initRandomArray(cpu_output_gradients, total_size);
  initRandomArray(cpu_weights, param_size);
  // printArray(cpu_input, size, size, "input batch 0");
  // printArray(cpu_input + size * size, size, size, "input batch 1");
  // printArray(cpu_weights, channels, 2, "weights");
  // printArray(cpu_output_gradients, size, size, "batch 0 output gradients");
  // printArray(cpu_output_gradients + size * size,
  //            size,
  //            size,
  //            "batch 1 output gradients");
  float *gpu_output_gradients = cudaGetPointer<float>(total_size);
  float *gpu_input = cudaGetPointer<float>(total_size);
  float *gpu_batch_stats = cudaGetPointer<float>(param_size);
  float *gpu_weights = cudaGetPointer<float>(param_size);
  float *gpu_input_gradients = cudaGetPointer<float>(total_size);
  float *gpu_weight_gradients = cudaGetPointer<float>(param_size);
  float *gpu_normalized_output = cudaGetPointer<float>(total_size);
  float *gpu_running_stats = cudaGetPointer<float>(param_size);
  cudaLibCopyToDevice(cpu_input, gpu_input, total_size);
  cudaLibCopyToDevice(cpu_output_gradients, gpu_output_gradients, total_size);
  cudaLibCopyToDevice(cpu_weights, gpu_weights, param_size);
  cudaBatchNormTrain(gpu_input,
                     gpu_normalized_output,
                     gpu_weights,
                     gpu_batch_stats,
                     gpu_running_stats,
                     channels,
                     size,
                     size,
                     batch_size);

  float *cpu_batch_stats = cudaToCPU(gpu_batch_stats, param_size);
  float *cpu_normalized_output = cudaToCPU(gpu_normalized_output, total_size);

  // printArray(cpu_batch_stats, channels, 2, "batch stats");
  // printArray(cpu_normalized_output, size, size, "output batch 0");
  // printArray(cpu_normalized_output + size * size, size, size, "output batch
  // 1");

  cpuBatchNormBackwards(cpu_input,
                        cpu_output_gradients,
                        cpu_weights,
                        cpu_batch_stats,
                        cpu_input_gradients,
                        cpu_weight_gradients,
                        channels,
                        batch_size,
                        size);

  cudaBatchNormBackward(gpu_output_gradients,
                        gpu_input,
                        gpu_batch_stats,
                        gpu_weights,
                        gpu_input_gradients,
                        gpu_weight_gradients,
                        channels,
                        size,
                        size,
                        batch_size);
  float *weight_gradients_host = cudaToCPU(gpu_weight_gradients, param_size);
  float *input_gradients_host = cudaToCPU(gpu_input_gradients, total_size);
  // printArray(weight_gradients_host, channels, 2, "gpu weight gradients");
  // printArray(input_gradients_host, size, size, "gpu input gradients batch
  // 0"); printArray(input_gradients_host + size * size,
  //            size,
  //            size,
  //            "gpu input gradients batch 1");
  // printArray(cpu_weight_gradients, channels, 2, "cpu weight gradients");
  // printArray(cpu_input_gradients, size, size, "cpu input gradients batch 0");
  // printArray(cpu_input_gradients + size * size,
  //            size,
  //            size,
  //            "cpu input gradients batch 1");

  assert(checkArrayEquivalence(
      input_gradients_host, cpu_input_gradients, total_size));
  assert(checkArrayEquivalence(
      weight_gradients_host, cpu_weight_gradients, param_size));

  delete[] cpu_input;
  delete[] cpu_output_gradients;
  delete[] cpu_weights;
  delete[] cpu_input_gradients;
  delete[] cpu_weight_gradients;
  delete[] cpu_batch_stats;
  delete[] cpu_normalized_output;
  delete[] weight_gradients_host;
  delete[] input_gradients_host;

  cudaLibFree(gpu_output_gradients);
  cudaLibFree(gpu_input);
  cudaLibFree(gpu_batch_stats);
  cudaLibFree(gpu_weights);
  cudaLibFree(gpu_input_gradients);
  cudaLibFree(gpu_weight_gradients);
  cudaLibFree(gpu_normalized_output);
  cudaLibFree(gpu_running_stats);
}

inline void UNetTest::testBatchNorm() {

  testBatchNormBackwardTraining();
  testBatchNormForwardInference();
  testBatchNormForwardTraining();

  std::cout << "\033[32mBatch Norm tests pass!\033[0m" << std::endl;
}
