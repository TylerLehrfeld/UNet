#include "cuda_lib.h"
#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void XAVIER_or_HE_initialize(float *weights, int sqrt_N,
                                        int num_weights) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_weights)
    return;
  curandState state;
  unsigned long seed = 0;
  curand_init(seed, idx, 0, &state); // idx makes streams unique
  weights[idx] = curand_normal(&state) * 1.41421356237 / sqrt_N;
}

float *getCudaPointer(int num_floats, Initiation_type i_type, int N,
                      int num_weights) {
  float *dPointer;
  cudaMalloc(&dPointer, num_floats * sizeof(float));
  cudaMemset(dPointer, 0, num_floats * sizeof(float));
  if (i_type == XAVIER || i_type == HE) {
    dim3 blockDim(128);
    dim3 gridDim((num_weights + blockDim.x - 1) / blockDim.x);
    XAVIER_or_HE_initialize<<<gridDim, blockDim>>>(dPointer, sqrt(N),
                                                   num_weights);
    cudaDeviceSynchronize();
  }
  return dPointer;
}

__device__ inline int flattened_index(int batch_num, int channel,
                                      int num_channels, int h, int H, int w,
                                      int W) {
  return batch_num * (num_channels * H * W) + channel * (H * W) + h * W + w;
}

__global__ void convolution(const float *weights, const float *inputs,
                            float *activations, int activation_array_length,
                            int batch_size, int kernel_size, int channels_in,
                            int channels_out, int H_out, int W_out, int H_in,
                            int W_in, int padding, int stride) {
  int kern_squared = kernel_size * kernel_size;
  int idx = threadIdx.x + blockIdx.x * blockDim.x +
            (threadIdx.y + blockIdx.y * blockDim.y) * (gridDim.x * blockDim.x);

  // int batch_num = idx / W_out / H_out / channels_out;
  // int out_channel = ((idx / W_out / H_out) % channels_out);
  // int h_out = (idx / W_out) % H_out;
  // int w_out = idx % W_out;
  int spatial_idx = threadIdx.x + blockIdx.y * blockDim.x;
  int batch_num = blockIdx.z;
  int out_channel = blockIdx.x;
  int h_out = spatial_idx / W_out;
  int w_out = spatial_idx % W_out;
  int input_i = h_out * stride - padding + (kernel_size - 1) / 2;
  int input_j = w_out * stride - padding + (kernel_size - 1) / 2;
  extern __shared__ float local_kernels[];
  int block_size = blockDim.x * blockDim.y;
  for (int i = threadIdx.y * blockDim.x + threadIdx.x;
       i < channels_in * kernel_size * kernel_size; i += block_size) {
    int c = i / (kern_squared);
    int k = i % kern_squared;
    local_kernels[c * kern_squared + k] =
        weights[flattened_index(c, out_channel, channels_out, k / kernel_size,
                                kernel_size, k % kernel_size, kernel_size)];
  }
  __syncthreads();
  if (idx >= activation_array_length)
    return;
  float sum = 0;
#pragma unroll
  for (int c = 0; c < channels_in; c++) {
#pragma unroll
    for (int i = -kernel_size / 2; i <= kernel_size / 2; i++) {
#pragma unroll
      for (int j = -kernel_size / 2; j <= kernel_size / 2; j++) {
        int i_ind = input_i + i;
        int j_ind = input_j + j;
        if (j_ind >= 0 && i_ind >= 0 && j_ind < W_in && i_ind < H_in) {
          int k = (i + kernel_size / 2) * kernel_size + j + kernel_size / 2;
          float weight = local_kernels[c * kernel_size * kernel_size + k];
          // We index weights as follows weights[in_channel][out_channel][i][j]
          sum +=
              weight *
              // weights[flattened_index(c, out_channel, channels_out,
              //                         i + kernel_size / 2, kernel_size,
              //                         j + kernel_size / 2, kernel_size)] *
              //  We index input images as follows: inputs[batch][channel][i][j]
              inputs[flattened_index(batch_num, c, channels_in, i_ind, H_in,
                                     j_ind, W_in)];
        }
      }
    }
  }
  // after the weights (convolution kernels), there are biases
  sum += weights[channels_out * channels_in * kernel_size * kernel_size +
                 out_channel];
  // we index outpus like inputs: activations[batch][channel][i][j]
  activations[flattened_index(batch_num, out_channel, channels_out, h_out,
                              H_out, w_out, W_out)] = sum;
}

void convolve(const float *weights, const float *inputs, float *activations,
              int activation_array_length, int batch_size, int kernel_size,
              int channels_in, int channels_out, int H_out, int W_out, int H_in,
              int W_in, int padding, int stride) {
  int out_image_size = W_out * H_out;
  dim3 blockDim;
  dim3 gridDim;

  if (out_image_size > 1024) {
    blockDim.x = 1024;
    blockDim.y = 1;
    gridDim.x = channels_out;
    gridDim.y = out_image_size / 1024;
    gridDim.z = batch_size;
  } else {
    blockDim.x = out_image_size;
    blockDim.y = 1;
    gridDim.x = channels_out;
    gridDim.y = 1;
    gridDim.z = 1;
  }

  int shared_memory_size =
      kernel_size * kernel_size * channels_out * sizeof(float);

  convolution<<<gridDim, blockDim, shared_memory_size>>>(
      weights, inputs, activations, activation_array_length, batch_size,
      kernel_size, channels_in, channels_out, H_out, W_out, H_in, W_in, padding,
      stride);
}
