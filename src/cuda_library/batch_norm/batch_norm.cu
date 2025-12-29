#include "../cuda_lib.h"
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

const float eps = 1e-8f;

template <typename T>
__global__ void BatchNormInferenceKernel(const T *__restrict__ inputs,
                                         T *__restrict__ activations,
                                         const T *__restrict__ weights,
                                         const T *__restrict__ stats,
                                         int H,
                                         int W,
                                         int num_channels) {
  int spatial_idx = threadIdx.x + blockIdx.y * blockDim.x;
  if(spatial_idx >= H * W) {
    return;
  }
  int out_channel = blockIdx.x;
  int h_out = spatial_idx / W;
  int w_out = spatial_idx % W;
  T bias = weights[out_channel * 2 + 1];
  T weight = weights[out_channel * 2];
  int idx = flattenIndex(0, out_channel, num_channels, h_out, H, w_out, W);
  activations[idx] = weight * ((inputs[idx] - stats[out_channel * 2]) /
                               sqrt(stats[out_channel * 2 + 1] + eps)) +
                     bias;
};

// BN stats have been PROCESSED. They are mean and variance now. Should probably
// be changed to mean and std dev.
template <typename T>
void cudaBatchNormInference(const T *__restrict__ inputs,
                            T *__restrict__ activations,
                            const T *__restrict__ weights,
                            const T *__restrict__ BN_stats,
                            int H,
                            int W,
                            int num_channels) {

  int out_image_size = H * W;
  dim3 blockDim;
  dim3 gridDim;

  if(out_image_size > 1024) {
    blockDim.x = 1024;
    blockDim.y = 1;
    gridDim.x = num_channels;
    gridDim.y = out_image_size / 1024;
    if(out_image_size % 1024 != 0)
      gridDim.y++;
  } else {
    blockDim.x = out_image_size;
    blockDim.y = 1;
    gridDim.x = num_channels;
    gridDim.y = 1;
  }
  BatchNormInferenceKernel<<<gridDim, blockDim>>>(
      inputs, activations, weights, BN_stats, H, W, num_channels);

  cudaDeviceSynchronize();
}

template void cudaBatchNormInference<float>(const float *__restrict__ inputs,
                                            float *__restrict__ activations,
                                            const float *__restrict__ weights,
                                            const float *__restrict__ BN_stats,
                                            int H,
                                            int W,
                                            int num_channels);

template <typename T>
__global__ void BatchNormTrainStatsKernel(const T *__restrict__ inputs,
                                          T *__restrict__ BN_batch_stats,
                                          int num_channels,
                                          int H,
                                          int W) {
  int channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H * W)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W;
  int w = spatial_idx % W;
  T val = inputs[flattenIndex(batch_idx, channel, num_channels, h, H, w, W)];

  // TODO:Make this WAY faster. Change to actual mean and variance. Maybe add
  // another array for the sums to accumulate gradients like these
  atomicAdd(&BN_batch_stats[channel * 2], val);
  atomicAdd(&BN_batch_stats[channel * 2 + 1], val * val);
}

template <typename T>
__global__ void
BatchNormTrainNormalizeKernel(const T *__restrict__ inputs,
                              T *__restrict__ activations,
                              const T *__restrict__ BN_batch_stats,
                              T *__restrict__ BN_running_stats,
                              const T *__restrict__ weights,
                              int H,
                              int W,
                              int num_channels,
                              int batch_size) {

  int channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H * W)
    return;
  int batch_idx = blockIdx.y;

  int h = spatial_idx / W;
  int w = spatial_idx % W;
  T mean = BN_batch_stats[channel * 2] / (H * W * batch_size);
  T var =
      (BN_batch_stats[channel * 2 + 1] / (H * W * batch_size)) - (mean * mean);
  // if(batch_idx == 0 && channel == 0 && h == 0 && w == 0) {
  //   printf("mean: %f, variance: %f\n", mean, var);
  // }
  T weight = weights[channel * 2];
  T bias = weights[channel * 2 + 1];
  activations[flattenIndex(batch_idx, channel, num_channels, h, H, w, W)] =
      weight *
          (inputs[flattenIndex(batch_idx, channel, num_channels, h, H, w, W)] -
           mean) /
          (sqrt(var) + eps) +
      bias;
  if(spatial_idx == 0 && batch_idx == 0) {
    BN_running_stats[channel * 2] =
        BN_running_stats[channel * 2] * kMomentum + kInverse_momentum * mean;
    BN_running_stats[channel * 2 + 1] =
        BN_running_stats[channel * 2 + 1] * kMomentum + kInverse_momentum * var;
  }
}

template <typename T>
void cudaBatchNormTrain(const T *__restrict__ inputs,
                        T *__restrict__ activations,
                        const T *__restrict__ weights,
                        T *__restrict__ BN_batch_stats,
                        T *__restrict__ BN_running_stats,
                        int num_channels,
                        int H,
                        int W,
                        int batch_size) {

  // TODO: Do this in a better way. Probably through cuda manager
  cudaMemset(BN_batch_stats, 0, num_channels * 2 * sizeof(T));
  int block_size = 768;
  dim3 gridDim(H * W / block_size, batch_size, num_channels);
  if(H * W % block_size != 0)
    gridDim.x++;
  dim3 blockDim(1024, 1);
  BatchNormTrainStatsKernel<<<gridDim, blockDim>>>(
      inputs, BN_batch_stats, num_channels, H, W);

  cudaDeviceSynchronize();

  BatchNormTrainNormalizeKernel<<<gridDim, blockDim>>>(inputs,
                                                       activations,
                                                       BN_batch_stats,
                                                       BN_running_stats,
                                                       weights,
                                                       H,
                                                       W,
                                                       num_channels,
                                                       batch_size);

  cudaDeviceSynchronize();
}

template void cudaBatchNormTrain<float>(const float *__restrict__ inputs,
                                        float *__restrict__ activations,
                                        const float *__restrict__ weights,
                                        float *__restrict__ BN_batch_stats,
                                        float *__restrict__ BN_running_stats,
                                        int num_channels,
                                        int H,
                                        int W,
                                        int batch_size);

template <typename T>
__global__ void BatchNormBackwardWeightsKernel(
    const T *grad_out,
    const T *inputs,
    T *grad_weights,         // [2*num_channels] = gamma, beta
    const T *BN_batch_stats, // [2*num_channels]
    int H,
    int W,
    int batch_size,
    int num_channels) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= num_channels)
    return;

  T grad_gamma = 0.0f;
  T grad_beta = 0.0f;

  int N = batch_size * H * W;
  T channel_mean = BN_batch_stats[c * 2] / N;
  T channel_var = BN_batch_stats[c * 2 + 1] / N - channel_mean * channel_mean;
  T channel_std = sqrt(channel_var + eps);
  if(isnan(channel_std)) {
    printf("channel %d std is nan. channel sum: %f, channel sum squared %f\n",
           c,
           BN_batch_stats[c * 2],
           BN_batch_stats[c * 2 + 1]);
  }
  for(int n = 0; n < batch_size; n++) {
    for(int h = 0; h < H; h++) {
      for(int w = 0; w < W; w++) {
        int idx = n * num_channels * H * W + c * H * W + h * W + w;

        T x_hat = (inputs[idx] - channel_mean) / channel_std;
        T g = grad_out[idx];
        grad_gamma += g * x_hat;
        grad_beta += g;
      }
    }
  }

  grad_weights[2 * c] = grad_gamma;
  grad_weights[2 * c + 1] = grad_beta;
}

template <typename T>
__global__ void BatchNormBackwardInputKernel(
    const T *__restrict__ grad_out,
    const T *__restrict__ inputs,
    const T *__restrict__ BN_batch_stats,
    const T *__restrict__ weights,      // gamma, beta
    const T *__restrict__ grad_weights, // sums of dy and dy*x_hat
    T *__restrict__ grad_input,
    int H,
    int W,
    int batch_size,
    int num_channels) {
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = blockIdx.y;
  int batch_idx = blockIdx.z;

  if(batch_idx >= batch_size || channel >= num_channels || spatial_idx >= H * W)
    return;

  int h = spatial_idx / W;
  int w = spatial_idx % W;
  int idx = batch_idx * num_channels * H * W + channel * H * W + h * W + w;

  int N = batch_size * H * W;
  T gamma = weights[2 * channel];
  T mean = BN_batch_stats[2 * channel] / N;
  T variance = BN_batch_stats[2 * channel + 1] / N - (mean * mean);

  T x_hat = (inputs[idx] - mean) / (sqrt(variance + eps));
  T grad_out_val = grad_out[idx];

  // Standard BN backward formula:
  T grad_input_val =
      gamma / sqrt(variance + eps) *

      (grad_out_val - grad_weights[2 * channel + 1] / N // grad_beta term
       - x_hat * grad_weights[2 * channel] / N          // grad_gamma term
      );

  grad_input[idx] = grad_input_val;
}

template <typename T>
void cudaBatchNormBackward(const T *__restrict__ grad_activations,
                           const T *__restrict__ inputs,
                           const T *__restrict__ BN_batch_stats,
                           const T *__restrict__ weights,
                           T *__restrict__ grad_inputs,
                           T *__restrict__ grad_weights,
                           int num_channels,
                           int H,
                           int W,
                           int batch_size) {

  // Compute grad_gamma and grad_beta
  dim3 block_gamma(1024);
  dim3 grid_gamma((num_channels + 1023) / 1024);

  // WARNING:BN_batch stats sometimes has negative variances: explain.
  BatchNormBackwardWeightsKernel<<<grid_gamma, block_gamma>>>(grad_activations,
                                                              inputs,
                                                              grad_weights,
                                                              BN_batch_stats,
                                                              H,
                                                              W,
                                                              batch_size,
                                                              num_channels);

  cudaDeviceSynchronize();
  // Compute grad_input
  dim3 block_input(1024);
  dim3 grid_input((H * W + 1023) / 1024, num_channels, batch_size);
  BatchNormBackwardInputKernel<<<grid_input, block_input>>>(grad_activations,
                                                            inputs,
                                                            BN_batch_stats,
                                                            weights,
                                                            grad_weights,
                                                            grad_inputs,
                                                            H,
                                                            W,
                                                            batch_size,
                                                            num_channels);

  cudaDeviceSynchronize();
}

template void
cudaBatchNormBackward<float>(const float *__restrict__ grad_activations,
                             const float *__restrict__ inputs,
                             const float *__restrict__ BN_batch_stats,
                             const float *__restrict__ weights,
                             float *__restrict__ grad_inputs,
                             float *__restrict__ grad_weights,
                             int num_channels,
                             int H,
                             int W,
                             int batch_size);
