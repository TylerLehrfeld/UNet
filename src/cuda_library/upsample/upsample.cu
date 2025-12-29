#include "../cuda_lib.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

template <typename T>
__global__ void upsampleKernel(const T *__restrict__ inputs,
                               const T *__restrict__ weights,
                               T *__restrict__ activations,
                               int scale,
                               int H_in,
                               int W_in,
                               int channels_in,
                               int channels_out,
                               int activation_array_length) {
  int total_magnification = scale * scale;
  int idx = threadIdx.x + blockIdx.x * blockDim.x +
            (threadIdx.y + blockIdx.y * blockDim.y) * (gridDim.x * blockDim.x);
  int spatial_idx = threadIdx.x + blockIdx.y * blockDim.x;
  int batch_num = blockIdx.z;
  int out_channel = blockIdx.x;
  int H_out = H_in * scale;
  int W_out = W_in * scale;
  int h_out = spatial_idx / W_out;
  int w_out = spatial_idx % W_out;
  int input_i = h_out / scale;
  int input_j = w_out / scale;
  extern __shared__ T local_kernels[];
  int block_size = blockDim.x * blockDim.y;
  for(int i = threadIdx.y * blockDim.x + threadIdx.x;
      i < channels_in * scale * scale;
      i += block_size) {
    int c = i / (total_magnification);
    int k = i % total_magnification;
    local_kernels[c * total_magnification + k] = weights[flattenIndex(
        c, out_channel, channels_out, k / scale, scale, k % scale, scale)];
  }
  __syncthreads();
  if(idx >= activation_array_length)
    return;
  T sum = 0;
#pragma unroll
  for(int c = 0; c < channels_in; c++) {
    T weight = local_kernels[c * total_magnification + scale * (h_out % scale) +
                             (w_out % scale)];
    T input_c = inputs[flattenIndex(
        batch_num, c, channels_in, input_i, H_in, input_j, W_in)];
    sum += weight * input_c;
  }
  // after the weights (convolution kernels), there are biases
  sum +=
      weights[channels_out * channels_in * total_magnification + out_channel];
  // we index outpus like inputs: activations[batch][channel][i][j]
  activations[flattenIndex(
      batch_num, out_channel, channels_out, h_out, H_out, w_out, W_out)] = sum;
}

template <typename T>
void cudaUpsample(const T *__restrict__ inputs,
                  const T *__restrict__ weights,
                  T *__restrict__ activations,
                  int scale,
                  int num_in_channels,
                  int num_out_channels,
                  int H_in,
                  int W_in,
                  int batch_size) {
  int out_image_size = scale * scale * W_in * H_in;
  dim3 blockDim;
  dim3 gridDim;

  if(out_image_size > 1024) {
    blockDim.x = 1024;
    blockDim.y = 1;
    gridDim.x = num_out_channels;
    gridDim.y = out_image_size / 1024;
    if(out_image_size % 1024 != 0)
      gridDim.y++;
    gridDim.z = batch_size;
  } else {
    blockDim.x = out_image_size;
    blockDim.y = 1;
    gridDim.x = num_out_channels;
    gridDim.y = 1;
    gridDim.z = batch_size;
  }

  int shared_memory_size = scale * scale * num_in_channels * sizeof(T);
  upsampleKernel<<<gridDim, blockDim, shared_memory_size>>>(
      inputs,
      weights,
      activations,
      scale,
      H_in,
      W_in,
      num_in_channels,
      num_out_channels,
      out_image_size * batch_size * num_out_channels);

  cudaDeviceSynchronize();
}

// Explicit template declaration for floats
template void cudaUpsample<float>(const float *__restrict__ inputs,
                                  const float *__restrict__ weights,
                                  float *__restrict__ activations,
                                  int scale,
                                  int num_in_channels,
                                  int num_out_channels,
                                  int H_in,
                                  int W_in,
                                  int batch_size);

template <typename T>
__global__ void
upsampleBackwardsInputKernel(const T *__restrict__ grad_activations,
                             const T *__restrict__ weights,
                             T *__restrict__ grad_inputs,
                             int scale,
                             int H_in,
                             int W_in,
                             int channels_in,
                             int channels_out) {
  int spatial_idx = threadIdx.x + blockIdx.y * blockDim.x;
  int batch_num = blockIdx.z;
  int in_channel = blockIdx.x;
  int h_in = spatial_idx / W_in;
  int w_in = spatial_idx % W_in;

  if(spatial_idx >= H_in * W_in)
    return;

  int H_out = H_in * scale;
  int W_out = W_in * scale;

  T grad_sum = 0.0f;
#pragma unroll
  for(int out_channel = 0; out_channel < channels_out; out_channel++) {
#pragma unroll
    for(int i = 0; i < scale; i++) {
#pragma unroll
      for(int j = 0; j < scale; j++) {
        int h_out = h_in * scale + i;
        int w_out = w_in * scale + j;

        T weight = weights[flattenIndex(
            in_channel, out_channel, channels_out, i, scale, j, scale)];
        T grad = grad_activations[flattenIndex(
            batch_num, out_channel, channels_out, h_out, H_out, w_out, W_out)];
        grad_sum += weight * grad;
      }
    }
  }

  grad_inputs[flattenIndex(
      batch_num, in_channel, channels_in, h_in, H_in, w_in, W_in)] = grad_sum;
}

template <typename T>
__global__ void
upsampleBackwardWeightsKernel(const T *__restrict__ grad_activations,
                              const T *__restrict__ inputs,
                              T *__restrict__ grad_weights,
                              int scale,
                              int H_in,
                              int W_in,
                              int channels_in,
                              int channels_out,
                              int batch_size) {
  int in_channel = blockIdx.x;
  int out_channel = blockIdx.y;
  int k_idx = blockIdx.z * blockDim.x + threadIdx.x;

  int total_magnification = scale * scale;
  if(k_idx >= total_magnification)
    return;

  int ki = k_idx / scale;
  int kj = k_idx % scale;

  int H_out = H_in * scale;
  int W_out = W_in * scale;

  T grad_sum = 0.0f;
#pragma unroll
  for(int batch = 0; batch < batch_size; batch++) {
    for(int h_in = 0; h_in < H_in; h_in++) {
      for(int w_in = 0; w_in < W_in; w_in++) {
        int h_out = h_in * scale + ki;
        int w_out = w_in * scale + kj;

        T grad = grad_activations[flattenIndex(
            batch, out_channel, channels_out, h_out, H_out, w_out, W_out)];
        T input = inputs[flattenIndex(
            batch, in_channel, channels_in, h_in, H_in, w_in, W_in)];
        grad_sum += grad * input;
      }
    }
  }

  int weight_idx =
      flattenIndex(in_channel, out_channel, channels_out, ki, scale, kj, scale);
  atomicAdd(&grad_weights[weight_idx], grad_sum);
}

template <typename T>
__global__ void
upsampleBackwardBiasKernel(const T *__restrict__ grad_activations,
                           T *__restrict__ grad_weights,
                           int H_out,
                           int W_out,
                           int channels_out,
                           int batch_size,
                           int weight_offset) {
  int out_channel = blockIdx.x;

  T grad_sum = 0.0f;
  for(int batch = 0; batch < batch_size; batch++) {
    for(int h = 0; h < H_out; h++) {
      for(int w = 0; w < W_out; w++) {
        grad_sum += grad_activations[flattenIndex(
            batch, out_channel, channels_out, h, H_out, w, W_out)];
      }
    }
  }

  atomicAdd(&grad_weights[weight_offset + out_channel], grad_sum);
}

template <typename T>
void cudaUpsampleBackward(const T *__restrict__ grad_activations,
                          const T *__restrict__ inputs,
                          const T *__restrict__ weights,
                          T *__restrict__ grad_inputs,
                          T *__restrict__ grad_weights,
                          int scale,
                          int channels_in,
                          int channels_out,
                          int H_in,
                          int W_in,
                          int batch_size) {
  int total_magnification = scale * scale;
  int H_out = H_in * scale;
  int W_out = W_in * scale;

  cudaMemset(grad_weights,
             0,
             (channels_in * channels_out * total_magnification + channels_out) *
                 sizeof(T));

  // Backward for inputs
  int in_image_size = H_in * W_in;
  dim3 blockDim_input(1024, 1);
  dim3 gridDim_input(channels_in, (in_image_size + 1023) / 1024, batch_size);

  upsampleBackwardsInputKernel<<<gridDim_input, blockDim_input>>>(
      grad_activations,
      weights,
      grad_inputs,
      scale,
      H_in,
      W_in,
      channels_in,
      channels_out);

  // Backward for weights
  dim3 blockDim_weight(1024, 1);
  dim3 gridDim_weight(
      channels_in, channels_out, (total_magnification + 1023) / 1024);

  upsampleBackwardWeightsKernel<<<gridDim_weight, blockDim_weight>>>(
      grad_activations,
      inputs,
      grad_weights,
      scale,
      H_in,
      W_in,
      channels_in,
      channels_out,
      batch_size);

  // Backward for bias
  dim3 gridDim_bias(channels_out, 1);
  dim3 blockDim_bias(1, 1);

  upsampleBackwardBiasKernel<<<gridDim_bias, blockDim_bias>>>(
      grad_activations,
      grad_weights,
      H_out,
      W_out,
      channels_out,
      batch_size,
      channels_in * channels_out * total_magnification);

  cudaDeviceSynchronize();
}

template void
cudaUpsampleBackward<float>(const float *__restrict__ grad_activations,
                            const float *__restrict__ inputs,
                            const float *__restrict__ weights,
                            float *__restrict__ grad_inputs,
                            float *__restrict__ grad_weights,
                            int scale,
                            int channels_in,
                            int channels_out,
                            int H_in,
                            int W_in,
                            int batch_size);
