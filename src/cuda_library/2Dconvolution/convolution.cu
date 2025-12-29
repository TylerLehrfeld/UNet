#include "../cuda_lib.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

template <typename T>
__global__ void convolutionKernel(const T *__restrict__ weights,
                                  const T *__restrict__ inputs,
                                  T *__restrict__ activations,
                                  int activation_array_length,
                                  int batch_size,
                                  int kernel_size,
                                  int channels_in,
                                  int channels_out,
                                  int H_out,
                                  int W_out,
                                  int H_in,
                                  int W_in,
                                  int padding,
                                  int stride) {
  int kern_squared = kernel_size * kernel_size;

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
  extern __shared__ T local_kernels[];
  int block_size = blockDim.x * blockDim.y;
  for(int i = threadIdx.y * blockDim.x + threadIdx.x;
      i < channels_in * kernel_size * kernel_size;
      i += block_size) {
    int c = i / (kern_squared);
    int k = i % kern_squared;
    int index =
        c * channels_out * kern_squared + out_channel * kern_squared + k;
    local_kernels[c * kern_squared + k] = weights[index];
  }
  __syncthreads();
  if(spatial_idx >= W_out * H_out)
    return;
  T sum = 0;
#pragma unroll
  for(int c = 0; c < channels_in; c++) {
#pragma unroll
    for(int i = -kernel_size / 2; i <= kernel_size / 2; i++) {
#pragma unroll
      for(int j = -kernel_size / 2; j <= kernel_size / 2; j++) {
        int i_ind = input_i + i;
        int j_ind = input_j + j;
        if(j_ind >= 0 && i_ind >= 0 && j_ind < W_in && i_ind < H_in) {
          int k = (i + kernel_size / 2) * kernel_size + j + kernel_size / 2;
          T weight = local_kernels[c * kernel_size * kernel_size + k];
          // We index weights as follows weights[in_channel][out_channel][i][j]
          sum +=
              weight *
              // weights[flattened_index(c, out_channel, channels_out,
              //                         i + kernel_size / 2, kernel_size,
              //                         j + kernel_size / 2, kernel_size)] *
              //  We index input images as follows: inputs[batch][channel][i][j]
              inputs[flattenIndex(
                  batch_num, c, channels_in, i_ind, H_in, j_ind, W_in)];
        }
      }
    }
  }
  // after the weights (convolution kernels), there are biases
  sum += weights[channels_out * channels_in * kernel_size * kernel_size +
                 out_channel];
  // we index outpus like inputs: activations[batch][channel][i][j]
  activations[flattenIndex(
      batch_num, out_channel, channels_out, h_out, H_out, w_out, W_out)] = sum;
}

template <typename T>
void cudaConvolve(const T *__restrict__ weights,
                  const T *__restrict__ inputs,
                  T *__restrict__ activations,
                  int activation_array_length,
                  int batch_size,
                  int kernel_size,
                  int channels_in,
                  int channels_out,
                  int H_out,
                  int W_out,
                  int H_in,
                  int W_in,
                  int padding,
                  int stride) {
  int out_image_size = W_out * H_out;
  dim3 blockDim;
  dim3 gridDim;

  if(out_image_size > 1024) {
    blockDim.x = 1024;
    blockDim.y = 1;
    gridDim.x = channels_out;
    gridDim.y = out_image_size / 1024;
    if(out_image_size % 1024 != 0)
      gridDim.y++;
    gridDim.z = batch_size;
  } else {
    blockDim.x = out_image_size;
    blockDim.y = 1;
    gridDim.x = channels_out;
    gridDim.y = 1;
    gridDim.z = batch_size;
  }

  int shared_memory_size = kernel_size * kernel_size * channels_in * sizeof(T);

  convolutionKernel<<<gridDim, blockDim, shared_memory_size>>>(
      weights,
      inputs,
      activations,
      activation_array_length,
      batch_size,
      kernel_size,
      channels_in,
      channels_out,
      H_out,
      W_out,
      H_in,
      W_in,
      padding,
      stride);
}

template void cudaConvolve<float>(const float *__restrict__,
                                  const float *__restrict__,
                                  float *__restrict__,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int);

template <typename T>
__global__ void
convolutionBackwardInputKernel(const T *__restrict__ grad_activations,
                               const T *__restrict__ weights,
                               T *__restrict__ grad_inputs,
                               int batch_size,
                               int kernel_size,
                               int channels_in,
                               int channels_out,
                               int H_out,
                               int W_out,
                               int H_in,
                               int W_in,
                               int padding,
                               int stride) {
  int spatial_idx = threadIdx.x + blockIdx.y * blockDim.x;
  int batch_num = blockIdx.z;
  int in_channel = blockIdx.x;
  int h_in = spatial_idx / W_in;
  int w_in = spatial_idx % W_in;

  if(spatial_idx >= H_in * W_in)
    return;

  T grad_sum = static_cast<T>(0.0);
  int kern_squared = kernel_size * kernel_size;

  // Load weights into shared memory
  extern __shared__ T local_weights[];
  for(int out_ch = threadIdx.x; out_ch < channels_out; out_ch += blockDim.x) {
    for(int k = 0; k < kern_squared; k++) {
      int index =
          in_channel * channels_out * kern_squared + out_ch * kern_squared + k;
      local_weights[out_ch * kern_squared + k] = weights[index];
    }
  }

  __syncthreads();

  for(int out_channel = 0; out_channel < channels_out; out_channel++) {
    for(int kh = 0; kh < kernel_size; kh++) {
      for(int kw = 0; kw < kernel_size; kw++) {
        int h_out = (h_in + padding - kh) / stride;
        int w_out = (w_in + padding - kw) / stride;

        if(h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out &&
           (h_in + padding - kh) % stride == 0 &&
           (w_in + padding - kw) % stride == 0) {

          int k = kh * kernel_size + kw;
          T weight = local_weights[out_channel * kern_squared + k];

          T grad = grad_activations[flattenIndex(batch_num,
                                                 out_channel,
                                                 channels_out,
                                                 h_out,
                                                 H_out,
                                                 w_out,
                                                 W_out)];
          grad_sum += weight * grad;
        }
      }
    }
  }

  grad_inputs[flattenIndex(
      batch_num, in_channel, channels_in, h_in, H_in, w_in, W_in)] = grad_sum;
}

template <typename T>
__global__ void
convolutionBackwardWeightsKernel(const T *__restrict__ grad_activations,
                                 const T *__restrict__ inputs,
                                 T *__restrict__ grad_weights,
                                 int batch_size,
                                 int kernel_size,
                                 int channels_in,
                                 int channels_out,
                                 int H_out,
                                 int W_out,
                                 int H_in,
                                 int W_in,
                                 int padding,
                                 int stride) {
  // Each block handles one weight kernel (in_channel, out_channel)
  int in_channel = blockIdx.x;
  int out_channel = blockIdx.y;
  int k_idx = threadIdx.x; // Thread handles one weight position

  if(k_idx >= kernel_size * kernel_size)
    return;

  int kh = k_idx / kernel_size;
  int kw = k_idx % kernel_size;
  int kern_squared = kernel_size * kernel_size;

  T grad_sum = static_cast<T>(0.0);

  for(int batch = 0; batch < batch_size; batch++) {
    for(int h_out = 0; h_out < H_out; h_out++) {
      for(int w_out = 0; w_out < W_out; w_out++) {
        int h_in = h_out * stride - padding + kh;
        int w_in = w_out * stride - padding + kw;

        if(h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
          T grad = grad_activations[flattenIndex(
              batch, out_channel, channels_out, h_out, H_out, w_out, W_out)];
          T input = inputs[flattenIndex(
              batch, in_channel, channels_in, h_in, H_in, w_in, W_in)];
          grad_sum += grad * input;
        }
      }
    }
  }

  int weight_idx = in_channel * channels_out * kern_squared +
                   out_channel * kern_squared + k_idx;
  atomicAdd(&grad_weights[weight_idx], grad_sum);
}

template <typename T>
__global__ void
convolutionBackwardBiasKernel(const T *__restrict__ grad_activations,
                              T *__restrict__ grad_weights,
                              int batch_size,
                              int channels_out,
                              int H_out,
                              int W_out,
                              int weight_offset) {
  int out_channel = blockIdx.x * blockDim.x + threadIdx.x;

  if(out_channel >= channels_out)
    return;

  T grad_sum = static_cast<T>(0.0);
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
void cudaConvolveBackward(const T *__restrict__ grad_activations,
                          const T *__restrict__ inputs,
                          const T *__restrict__ weights,
                          T *__restrict__ grad_inputs,
                          T *__restrict__ grad_weights,
                          int batch_size,
                          int kernel_size,
                          int channels_in,
                          int channels_out,
                          int H_out,
                          int W_out,
                          int H_in,
                          int W_in,
                          int padding,
                          int stride) {
  cudaMemset((void *)grad_weights,
             0.0,
             sizeof(T) *
                 (channels_in * channels_out * kernel_size * kernel_size +
                  channels_out));

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  // Initialize grad_weights to zero
  int total_weights =
      channels_in * channels_out * kernel_size * kernel_size + channels_out;

  // 1. Compute gradient w.r.t. inputs
  int in_image_size = H_in * W_in;
  dim3 blockDim_input(256);
  dim3 gridDim_input(channels_in,
                     (in_image_size + blockDim_input.x - 1) / blockDim_input.x,
                     batch_size);

  int shared_mem_size = channels_out * kernel_size * kernel_size * sizeof(T);

  convolutionBackwardInputKernel<<<gridDim_input,
                                   blockDim_input,
                                   shared_mem_size,
                                   stream1>>>(grad_activations,
                                              weights,
                                              grad_inputs,
                                              batch_size,
                                              kernel_size,
                                              channels_in,
                                              channels_out,
                                              H_out,
                                              W_out,
                                              H_in,
                                              W_in,
                                              padding,
                                              stride);

  // 2. Compute gradient w.r.t. weights
  int k_squared = kernel_size * kernel_size;
  dim3 blockDim_weight(k_squared, 1); // One thread per weight in the kernel
  dim3 gridDim_weight(channels_in, channels_out, 1);

  convolutionBackwardWeightsKernel<<<gridDim_weight,
                                     blockDim_weight,
                                     0,
                                     stream2>>>(grad_activations,
                                                inputs,
                                                grad_weights,
                                                batch_size,
                                                kernel_size,
                                                channels_in,
                                                channels_out,
                                                H_out,
                                                W_out,
                                                H_in,
                                                W_in,
                                                padding,
                                                stride);

  // 3. Compute gradient w.r.t. biases
  dim3 blockDim_bias(256, 1);
  dim3 gridDim_bias((channels_out + blockDim_bias.x - 1) / blockDim_bias.x, 1);

  convolutionBackwardBiasKernel<<<gridDim_bias, blockDim_bias, 0, stream3>>>(
      grad_activations,
      grad_weights,
      batch_size,
      channels_out,
      H_out,
      W_out,
      channels_in * channels_out * kernel_size * kernel_size);

  cudaDeviceSynchronize();

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
}

template void
cudaConvolveBackward<float>(const float *__restrict__ grad_activations,
                            const float *__restrict__ inputs,
                            const float *__restrict__ weights,
                            float *__restrict__ grad_inputs,
                            float *__restrict__ grad_weights,
                            int batch_size,
                            int kernel_size,
                            int channels_in,
                            int channels_out,
                            int H_out,
                            int W_out,
                            int H_in,
                            int W_in,
                            int padding,
                            int stride);
