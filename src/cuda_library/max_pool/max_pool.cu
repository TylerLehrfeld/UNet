#include "../cuda_lib.h"
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdexcept>

template <typename T>
__global__ void maxPoolKernel(const T *__restrict__ inputs,
                              T *__restrict__ activations,
                              int stride,
                              int kernel_size,
                              int in_height,
                              int in_width) {
  int batch = blockIdx.z;
  int channel = threadIdx.x;
  int num_channels = blockDim.x;
  int h_out = blockIdx.x * blockDim.y + threadIdx.y;
  int w_out = blockIdx.y * blockDim.z + threadIdx.z;
  if(h_out >= in_height / stride || w_out >= in_width / stride) {
    return;
  }
  int h_in = h_out * stride;
  int w_in = w_out * stride;
  float max = -INFINITY;
#pragma unroll
  for(int i = 0; i < kernel_size; i++) {
#pragma unroll
    for(int j = 0; j < kernel_size; j++) {
      int h_in_loop = h_in + i;
      int w_in_loop = w_in + j;
      if(h_in_loop < in_height && w_in_loop < in_width) {
        T val = inputs[flattenIndex(batch,
                                    channel,
                                    num_channels,
                                    h_in_loop,
                                    in_height,
                                    w_in_loop,
                                    in_width)];
        if(val > max) {
          max = val;
        }
      }
    }
  }
  activations[flattenIndex(batch,
                           channel,
                           num_channels,
                           h_out,
                           in_height / stride,
                           w_out,
                           in_width / stride)] = max;
}

template <typename T>
void cudaMaxPool(const T *__restrict__ inputs,
                 T *__restrict__ activations,
                 int batch_size,
                 int stride,
                 int kernel_size,
                 int in_channels,
                 int in_height,
                 int in_width) {
  if(in_channels > 1024) {
    throw std::runtime_error("Max pool cannot handle more than 1024 channels "
                             "right now. Will implement later");
  }
  assert(in_width % stride == 0);
  assert(in_height % stride == 0);
  int out_width = in_width / stride;
  int out_height = in_height / stride;
  // KEEP 768. Highest speed seen. Maximises warps per SM
  dim3 blockDim(in_channels, 768 / in_channels, 1);
  dim3 gridDim(out_height / blockDim.y, out_width / blockDim.z, batch_size);
  if(768 % in_channels != 0) {
    blockDim.y++;
  }
  if(out_height % blockDim.y != 0) {
    gridDim.x++;
  }
  if(out_width % blockDim.z != 0) {
    gridDim.y++;
  }

  maxPoolKernel<<<gridDim, blockDim>>>(
      inputs, activations, stride, kernel_size, in_height, in_width);
}
template void cudaMaxPool<float>(const float *__restrict__ inputs,
                                 float *__restrict__ activations,
                                 int batch_size,
                                 int stride,
                                 int kernel_size,
                                 int in_channels,
                                 int in_height,
                                 int in_width);

template <typename T>
__global__ void maxPoolBackwardKernel(const T *__restrict__ grad_activations,
                                      const T *__restrict__ inputs,
                                      const T *__restrict__ activations,
                                      T *__restrict__ grad_inputs,
                                      int stride,
                                      int in_height,
                                      int in_width) {
  int batch = blockIdx.z;
  int channel = blockIdx.y;
  int num_channels = gridDim.y;
  int w_in = threadIdx.x;
  int h_in = blockIdx.x * blockDim.y * blockDim.z + blockDim.z * threadIdx.y +
             threadIdx.z;

  if(h_in >= in_height || w_in >= in_width) {
    return;
  }

  int h_out = h_in / stride;
  int w_out = w_in / stride;

  int out_height = in_height / stride;
  int out_width = in_width / stride;

  extern __shared__ T local_maxes[];

  int shared_ind =
      ((h_in % (blockDim.y * blockDim.z)) / stride) * (in_width / stride) +
      w_out;
  // printf("H in: %d, W in: %d, blockIdx: %d, shared ind: %d\n",
  //        h_in,
  //        w_in,
  //        blockIdx.x,
  //        shared_ind);
  int output_ind = flattenIndex(
      batch, channel, num_channels, h_out, out_height, w_out, out_width);
  // printf("threadIdx: %d, %d, %d, shrared_ind: %d\n",
  //        threadIdx.x,
  //        threadIdx.y,
  //        threadIdx.z,
  //        shared_ind);

  if(h_in % stride == 0 && w_in % stride == 0) {
    local_maxes[2 * shared_ind] = activations[output_ind];
    local_maxes[2 * shared_ind + 1] = grad_activations[output_ind];
  }
  __syncthreads();
  T max_val = local_maxes[2 * shared_ind];
  T grad = local_maxes[2 * shared_ind + 1];
  int input_ind = flattenIndex(
      batch, channel, num_channels, h_in, in_height, w_in, in_width);
  T input = inputs[input_ind];
  // TODO: See if we can do this without a branch and if it makes a performance
  // difference
  // if(input == max_val) {
  //  grad_inputs[input_ind] = grad;
  //} else {
  //  grad_inputs[input_ind] = 0.0f;
  //}

  // printf("H in: %d, W in: %d, input: %f, output: %f, gradient out: %f, "
  //        "gradient in: %f, shared_ind: %d, input_ind: %d, output_ind: %d\n",
  //        h_in,
  //        w_in,
  //        input,
  //        max_val,
  //        grad,
  //        grad_inputs[input_ind],
  //        shared_ind,
  //        input_ind,
  //        output_ind);
  grad_inputs[input_ind] = (input == max_val) * grad;
}

template <typename T>
void cudaMaxPoolBackward(const T *__restrict__ grad_activations,
                         const T *__restrict__ inputs,
                         const T *__restrict__ activations,
                         T *__restrict__ grad_inputs,
                         int batch_size,
                         int stride,
                         int in_channels,
                         int in_height,
                         int in_width) {
  if(in_channels > 1024) {
    throw std::runtime_error(
        "Max pool backward cannot handle more than 1024 channels");
  }

  dim3 blockDim(in_width,
                std::min(768 / (stride * in_width), in_height / stride),
                stride);
  dim3 gridDim(in_height / (blockDim.y * stride), in_channels, batch_size);
  if(in_height % (blockDim.y * stride) != 0) {
    gridDim.x++;
  }
  if(blockDim.y == 0) {
    throw std::runtime_error("Max pool backward cannot be done efficiently "
                             "when stride * in_width > 768");
  }
  // 2 numbers for each block. Each block occupies in_width / stride in x and y
  // is of height stride
  int shared_size =
      sizeof(T) * 2 * blockDim.x * blockDim.y * blockDim.z / (stride * stride);
  maxPoolBackwardKernel<<<gridDim, blockDim, shared_size>>>(grad_activations,
                                                            inputs,
                                                            activations,
                                                            grad_inputs,
                                                            stride,
                                                            in_height,
                                                            in_width);
}
template void cudaMaxPoolBackward<float>(const float *__restrict__,
                                         const float *__restrict__,
                                         const float *__restrict__,
                                         float *__restrict__,
                                         int,
                                         int,
                                         int,
                                         int,
                                         int);
