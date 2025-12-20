#include "cuda_lib.h"
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sstream>
#include <stdexcept>

const float MOMENTUM = 0.9f;
const float INV_MOMENTUM = 1 - MOMENTUM;

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void
XAVIER_or_HE_initialize(float *weights, float sqrt_N, int num_weights) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= num_weights)
    return;
  curandState state;
  unsigned long seed = 0;
  curand_init(seed, idx, 0, &state); // idx makes streams unique
  float norm_samp = curand_normal(&state);
  weights[idx] = norm_samp * 1.41421356237 / sqrt_N;
}

__host__ float *
getCudaPointer(int num_floats, Initiation_type i_type, int N, int num_weights) {
  float *dPointer;
  cudaMalloc(&dPointer, num_floats * sizeof(float));
  cudaMemset(dPointer, 0, num_floats * sizeof(float));
  if(i_type == XAVIER || i_type == HE) {
    dim3 blockDim(128);
    dim3 gridDim((num_weights + blockDim.x - 1) / blockDim.x);
    XAVIER_or_HE_initialize<<<gridDim, blockDim>>>(
        dPointer, sqrtf(N), num_weights);
    if(N == 0) {
      throw std::runtime_error("Can't pass N=0 to getCudaPointer when "
                               "initialzing with HE or XAVIER");
    }
    cudaDeviceSynchronize();
  } else if(i_type == BN) {
    float params_cpu[num_floats];
    for(int i = 0; i < num_floats / 2; i++) {
      params_cpu[2 * i] = 1.0f;
      params_cpu[2 * i + 1] = 0.0f;
    }
    cudaMemcpy(dPointer,
               params_cpu,
               num_floats * sizeof(float),
               cudaMemcpyHostToDevice);
  }
  return dPointer;
}

__device__ inline int flattened_index(
    int batch_num, int channel, int num_channels, int h, int H, int w, int W) {
  return batch_num * (num_channels * H * W) + channel * (H * W) + h * W + w;
}

template <typename T>
__global__ void convolution(const T *__restrict__ weights,
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
  extern __shared__ float local_kernels[];
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
              inputs[flattened_index(
                  batch_num, c, channels_in, i_ind, H_in, j_ind, W_in)];
        }
      }
    }
  }
  // after the weights (convolution kernels), there are biases
  sum += weights[channels_out * channels_in * kernel_size * kernel_size +
                 out_channel];
  // we index outpus like inputs: activations[batch][channel][i][j]
  activations[flattened_index(
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

  int shared_memory_size =
      kernel_size * kernel_size * channels_in * sizeof(float);

  convolution<<<gridDim, blockDim, shared_memory_size>>>(
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

__global__ void convolution_backward_input(float *grad_activations,
                                           float *weights,
                                           float *grad_inputs,
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

  float grad_sum = 0.0f;
  int kern_squared = kernel_size * kernel_size;

  // Load weights into shared memory
  extern __shared__ float local_weights[];
  if(threadIdx.x == 0) { // Only one thread per block loads weights
    for(int out_ch = 0; out_ch < channels_out; out_ch++) {
      for(int k = 0; k < kern_squared; k++) {
        int index = in_channel * channels_out * kern_squared +
                    out_ch * kern_squared + k;
        local_weights[out_ch * kern_squared + k] = weights[index];
      }
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
          float weight = local_weights[out_channel * kern_squared + k];

          float grad = grad_activations[flattened_index(batch_num,
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

  grad_inputs[flattened_index(
      batch_num, in_channel, channels_in, h_in, H_in, w_in, W_in)] = grad_sum;
}

__global__ void convolution_backward_weights(float *grad_activations,
                                             float *inputs,
                                             float *grad_weights,
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

  float grad_sum = 0.0f;

  for(int batch = 0; batch < batch_size; batch++) {
    for(int h_out = 0; h_out < H_out; h_out++) {
      for(int w_out = 0; w_out < W_out; w_out++) {
        int h_in = h_out * stride - padding + kh;
        int w_in = w_out * stride - padding + kw;

        if(h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
          float grad = grad_activations[flattened_index(
              batch, out_channel, channels_out, h_out, H_out, w_out, W_out)];
          float input = inputs[flattened_index(
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

__global__ void convolution_backward_bias(float *grad_activations,
                                          float *grad_weights,
                                          int batch_size,
                                          int channels_out,
                                          int H_out,
                                          int W_out,
                                          int weight_offset) {
  int out_channel = blockIdx.x * blockDim.x + threadIdx.x;

  if(out_channel >= channels_out)
    return;

  float grad_sum = 0.0f;
  for(int batch = 0; batch < batch_size; batch++) {
    for(int h = 0; h < H_out; h++) {
      for(int w = 0; w < W_out; w++) {
        grad_sum += grad_activations[flattened_index(
            batch, out_channel, channels_out, h, H_out, w, W_out)];
      }
    }
  }

  atomicAdd(&grad_weights[weight_offset + out_channel], grad_sum);
}

template <typename T>
void cudaConvolveBackward(T *grad_activations,
                          T *inputs,
                          T *weights,
                          T *grad_inputs,
                          T *grad_weights,
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

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  // Initialize grad_weights to zero
  int total_weights =
      channels_in * channels_out * kernel_size * kernel_size + channels_out;

  // 1. Compute gradient w.r.t. inputs
  int in_image_size = H_in * W_in;
  dim3 blockDim_input(256, 1);
  dim3 gridDim_input(channels_in,
                     (in_image_size + blockDim_input.x - 1) / blockDim_input.x,
                     batch_size);

  int shared_mem_size =
      channels_out * kernel_size * kernel_size * sizeof(float);

  convolution_backward_input<<<gridDim_input,
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

  convolution_backward_weights<<<gridDim_weight, blockDim_weight, 0, stream2>>>(
      grad_activations,
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

  convolution_backward_bias<<<gridDim_bias, blockDim_bias, 0, stream3>>>(
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

template void cudaConvolveBackward<float>(float *grad_activations,
                                          float *inputs,
                                          float *weights,
                                          float *grad_inputs,
                                          float *grad_weights,
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

void cuda_concat(float *activations_1,
                 float *activations_2,
                 float *activations,
                 int batch_size,
                 int HxW,
                 int C1,
                 int C2) {
  int C_tot = C1 + C2;
  for(int i = 0; i < batch_size; i++) {
    cudaMemcpy(activations + i * ((C_tot)*HxW),
               activations_1 + i * C1 * HxW,
               C1 * HxW * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(activations + i * (C_tot)*HxW + C1 * HxW,
               activations_2 + i * C2 * HxW,
               C2 * HxW * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
}
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
        T val = inputs[flattened_index(batch,
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
  activations[flattened_index(batch,
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
  int output_ind = flattened_index(
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
  int input_ind = flattened_index(
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

__global__ void upsample(const float *inputs,
                         const float *weights,
                         float *activations,
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
  extern __shared__ float local_kernels[];
  int block_size = blockDim.x * blockDim.y;
  for(int i = threadIdx.y * blockDim.x + threadIdx.x;
      i < channels_in * scale * scale;
      i += block_size) {
    int c = i / (total_magnification);
    int k = i % total_magnification;
    local_kernels[c * total_magnification + k] = weights[flattened_index(
        c, out_channel, channels_out, k / scale, scale, k % scale, scale)];
  }
  __syncthreads();
  if(idx >= activation_array_length)
    return;
  float sum = 0;
#pragma unroll
  for(int c = 0; c < channels_in; c++) {
    float weight = local_kernels[c * total_magnification +
                                 scale * (h_out % scale) + (w_out % scale)];
    float input_c = inputs[flattened_index(
        batch_num, c, channels_in, input_i, H_in, input_j, W_in)];
    sum += weight * input_c;
  }
  // after the weights (convolution kernels), there are biases
  sum +=
      weights[channels_out * channels_in * total_magnification + out_channel];
  // we index outpus like inputs: activations[batch][channel][i][j]
  activations[flattened_index(
      batch_num, out_channel, channels_out, h_out, H_out, w_out, W_out)] = sum;
}
void cuda_upsample(const float *inputs,
                   const float *weights,
                   float *activations,
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

  int shared_memory_size = scale * scale * num_in_channels * sizeof(float);
  upsample<<<gridDim, blockDim, shared_memory_size>>>(
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

__global__ void upsample_backward_input(const float *grad_activations,
                                        const float *weights,
                                        float *grad_inputs,
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

  float grad_sum = 0.0f;
#pragma unroll
  for(int out_channel = 0; out_channel < channels_out; out_channel++) {
#pragma unroll
    for(int i = 0; i < scale; i++) {
#pragma unroll
      for(int j = 0; j < scale; j++) {
        int h_out = h_in * scale + i;
        int w_out = w_in * scale + j;

        float weight = weights[flattened_index(
            in_channel, out_channel, channels_out, i, scale, j, scale)];
        float grad = grad_activations[flattened_index(
            batch_num, out_channel, channels_out, h_out, H_out, w_out, W_out)];
        grad_sum += weight * grad;
      }
    }
  }

  grad_inputs[flattened_index(
      batch_num, in_channel, channels_in, h_in, H_in, w_in, W_in)] = grad_sum;
}

__global__ void upsample_backward_weights(const float *grad_activations,
                                          const float *inputs,
                                          float *grad_weights,
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

  float grad_sum = 0.0f;
#pragma unroll
  for(int batch = 0; batch < batch_size; batch++) {
    for(int h_in = 0; h_in < H_in; h_in++) {
      for(int w_in = 0; w_in < W_in; w_in++) {
        int h_out = h_in * scale + ki;
        int w_out = w_in * scale + kj;

        float grad = grad_activations[flattened_index(
            batch, out_channel, channels_out, h_out, H_out, w_out, W_out)];
        float input = inputs[flattened_index(
            batch, in_channel, channels_in, h_in, H_in, w_in, W_in)];
        grad_sum += grad * input;
      }
    }
  }

  int weight_idx = flattened_index(
      in_channel, out_channel, channels_out, ki, scale, kj, scale);
  atomicAdd(&grad_weights[weight_idx], grad_sum);
}

__global__ void upsample_backward_bias(const float *grad_activations,
                                       float *grad_weights,
                                       int H_out,
                                       int W_out,
                                       int channels_out,
                                       int batch_size,
                                       int weight_offset) {
  int out_channel = blockIdx.x;

  float grad_sum = 0.0f;
  for(int batch = 0; batch < batch_size; batch++) {
    for(int h = 0; h < H_out; h++) {
      for(int w = 0; w < W_out; w++) {
        grad_sum += grad_activations[flattened_index(
            batch, out_channel, channels_out, h, H_out, w, W_out)];
      }
    }
  }

  atomicAdd(&grad_weights[weight_offset + out_channel], grad_sum);
}

void cuda_upsample_backward(const float *grad_activations,
                            const float *inputs,
                            const float *weights,
                            float *grad_inputs,
                            float *grad_weights,
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
                 sizeof(float));

  // Backward for inputs
  int in_image_size = H_in * W_in;
  dim3 blockDim_input(1024, 1);
  dim3 gridDim_input(channels_in, (in_image_size + 1023) / 1024, batch_size);

  upsample_backward_input<<<gridDim_input, blockDim_input>>>(grad_activations,
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

  upsample_backward_weights<<<gridDim_weight, blockDim_weight>>>(
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

  upsample_backward_bias<<<gridDim_bias, blockDim_bias>>>(
      grad_activations,
      grad_weights,
      H_out,
      W_out,
      channels_out,
      batch_size,
      channels_in * channels_out * total_magnification);

  cudaDeviceSynchronize();
}

__global__ void relu(float *inputs, float *activations, int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    float val = inputs[idx];
    if(val > 0) {
      activations[idx] = val;
    } else {
      activations[idx] = 0;
    }
  }
}

void cuda_relu(float *inputs, float *activations, int num_activations) {
  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  }
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  relu<<<gridDim, blockDim>>>(inputs, activations, num_activations);

  cudaDeviceSynchronize();
}

__global__ void
sigmoid(float *inputs, float *activations, int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    activations[idx] = sigmoid(inputs[idx]);
  }
}

void cuda_sigmoid(float *inputs, float *activations, int num_activations) {
  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  };
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  sigmoid<<<gridDim, blockDim>>>(inputs, activations, num_activations);

  cudaDeviceSynchronize();
}

__global__ void sigmoid_backward(float *grad_activations,
                                 float *activations,
                                 float *grad_inputs,
                                 int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    float sig = activations[idx];
    grad_inputs[idx] = grad_activations[idx] * sig * (1.0f - sig);
  }
}

void cuda_sigmoid_backward(float *grad_activations,
                           float *activations,
                           float *grad_inputs,
                           int num_activations) {
  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  }
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  sigmoid_backward<<<gridDim, blockDim>>>(
      grad_activations, activations, grad_inputs, num_activations);

  cudaDeviceSynchronize();
}

__global__ void relu_backward(float *grad_activations,
                              float *inputs,
                              float *grad_inputs,
                              int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    grad_inputs[idx] = (inputs[idx] > 0) ? grad_activations[idx] : 0.0f;
  }
}

void cuda_relu_backward(float *grad_activations,
                        float *inputs,
                        float *grad_inputs,
                        int num_activations) {
  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  }
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  relu_backward<<<gridDim, blockDim>>>(
      grad_activations, inputs, grad_inputs, num_activations);

  cudaDeviceSynchronize();
}

__global__ void BN_inference(float *inputs,
                             float *activations,
                             float *weights,
                             float *stats,
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
  float bias = weights[out_channel * 2 + 1];
  float weight = weights[out_channel * 2];
  int idx = flattened_index(1, out_channel, num_channels, h_out, H, w_out, W);
  activations[idx] = weight * ((inputs[idx] - stats[out_channel * 2]) /
                               stats[out_channel * 2 + 1]) +
                     bias;
};

void cuda_BN_inference(float *inputs,
                       float *activations,
                       float *weights,
                       float *BN_stats,
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
  BN_inference<<<gridDim, blockDim>>>(
      inputs, activations, weights, BN_stats, H, W, num_channels);

  cudaDeviceSynchronize();
}

__global__ void BN_train_stats(
    float *inputs, float *BN_batch_stats, int num_channels, int H, int W) {
  int channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H * W)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W;
  int w = spatial_idx % W;
  float val =
      inputs[flattened_index(batch_idx, channel, num_channels, h, H, w, W)];

  atomicAdd(&BN_batch_stats[channel * 2], val);
  atomicAdd(&BN_batch_stats[channel * 2 + 1], val * val);
}

__global__ void BN_train_normalize(float *inputs,
                                   float *activations,
                                   float *BN_batch_stats,
                                   float *BN_stats,
                                   float *weights,
                                   int H,
                                   int W,
                                   int num_channels,
                                   int batch_size) {

  float eps = 1e-8f;
  int channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H * W)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W;
  int w = spatial_idx % W;
  float mean = BN_batch_stats[channel * 2] / (H * W * batch_size);
  float var =
      (BN_batch_stats[channel * 2 + 1] - mean * mean) / (H * W * batch_size);
  float weight = weights[channel * 2];
  float bias = weights[channel * 2 + 1];
  activations[flattened_index(batch_idx, channel, num_channels, h, H, w, W)] =
      weight *
          (inputs[flattened_index(
               batch_idx, channel, num_channels, h, H, w, W)] -
           mean) /
          (sqrtf(var) + eps) +
      bias;
  if(spatial_idx == 0 && batch_idx == 0) {
    BN_stats[channel * 2] =
        BN_stats[channel * 2] * MOMENTUM + INV_MOMENTUM * mean;
    BN_stats[channel * 2 + 1] =
        BN_stats[channel * 2 + 1] * MOMENTUM + INV_MOMENTUM * var;
  }
}

void cuda_BN_train(float *inputs,
                   float *activations,
                   float *weights,
                   float *BN_batch_stats,
                   float *BN_stats,
                   int num_channels,
                   int H,
                   int W,
                   int batch_size) {
  dim3 gridDim(H * W / 1024, batch_size, num_channels);
  if(H * W % 1024 != 0)
    gridDim.x++;
  dim3 blockDim(1024, 1);
  BN_train_stats<<<gridDim, blockDim>>>(
      inputs, BN_batch_stats, num_channels, H, W);

  cudaDeviceSynchronize();

  BN_train_normalize<<<gridDim, blockDim>>>(inputs,
                                            activations,
                                            BN_batch_stats,
                                            BN_stats,
                                            weights,
                                            H,
                                            W,
                                            num_channels,
                                            batch_size);

  cudaDeviceSynchronize();
}
//__global__ void BN_backward(float *grad_activations,
//                            float *inputs,
//                            float *BN_batch_stats,
//                            float *weights,
//                            float *grad_inputs,
//                            float *grad_weights,
//                            int H,
//                            int W,
//                            int num_channels,
//                            int batch_size) {
//  int channel = blockIdx.z;
//  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if(spatial_idx >= H * W)
//    return;
//  int batch_idx = blockIdx.y;
//  int h = spatial_idx / W;
//  int w = spatial_idx % W;
//
//  int idx = flattened_index(batch_idx, channel, num_channels, h, H, w, W);
//
//  float mean = BN_batch_stats[channel * 2] / (H * W * batch_size);
//    float var =
//      (BN_batch_stats[channel * 2 + 1] / (H * W * batch_size)) - mean * mean;
//  float var_stable = var + 1e-5f;
//  float std = sqrt(var_stable);
//
//  float x = inputs[idx];
//  float x_norm = (x - mean) / std;
//  float gamma = weights[channel * 2];
//
//  float grad_out = grad_activations[idx];
//
//  atomicAdd(&grad_weights[channel * 2 + 1], grad_out);
//  atomicAdd(&grad_weights[channel * 2], grad_out * x_norm);
//
//  float N = H * W * batch_size;
//  float grad_x_norm = grad_out * gamma;
//
//  __shared__ float shared_grad_mean;
//  __shared__ float shared_grad_var;
//  if(threadIdx.x == 0) {
//    shared_grad_mean = 0.0f;
//    shared_grad_var = 0.0f;
//  }
//  __syncthreads();
//
//  atomicAdd(&shared_grad_mean, grad_x_norm);
//  atomicAdd(&shared_grad_var, grad_x_norm * (x - mean));
//  __syncthreads();
//
//  float dL_dx_norm_sum = shared_grad_mean;
//  float dL_dx_norm_x_mean_sum = shared_grad_var;
//
//  float term1 = N * grad_x_norm;
//  float term2 = dL_dx_norm_sum;
//  float term3 = dL_dx_norm_x_mean_sum * (x - mean) / var_stable;
//
//  grad_inputs[idx] = (term1 - term2 - term3) / (N * std);
//}

__global__ void BN_backward(float *grad_activations,
                            float *inputs,
                            float *BN_batch_stats,
                            float *weights,
                            float *grad_inputs,
                            float *grad_weights,
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

  int idx = flattened_index(batch_idx, channel, num_channels, h, H, w, W);

  float mean = BN_batch_stats[channel * 2] / (H * W * batch_size);
  float var =
      (BN_batch_stats[channel * 2 + 1] / (H * W * batch_size)) - mean * mean;
  float std = sqrt(var + 1e-5f);

  float x = inputs[idx];
  float x_norm = (x - mean) / std;
  float gamma = weights[channel * 2];

  float grad_out = grad_activations[idx];

  atomicAdd(&grad_weights[channel * 2 + 1], grad_out);
  atomicAdd(&grad_weights[channel * 2], grad_out * x_norm);

  float N = H * W * batch_size;
  float grad_x_norm = grad_out * gamma;

  __shared__ float shared_grad_mean;
  __shared__ float shared_grad_var;
  if(threadIdx.x == 0 && batch_idx == 0) {
    shared_grad_mean = 0.0f;
    shared_grad_var = 0.0f;
  }
  __syncthreads();

  atomicAdd(&shared_grad_mean, grad_x_norm);
  atomicAdd(&shared_grad_var, grad_x_norm * (x - mean));
  __syncthreads();

  float grad_mean = -shared_grad_mean / (std * N);
  float grad_var = -shared_grad_var / (2.0f * var * std * N);

  grad_inputs[idx] =
      grad_x_norm / std + grad_var * 2.0f * (x - mean) / N + grad_mean / N;
}

__global__ void
BN_backward_gamma_beta(const float *grad_out,
                       const float *inputs,
                       float *grad_weights, // [2*num_channels] = gamma, beta
                       const float *BN_batch_stats, // [num_channels]
                       int H,
                       int W,
                       int batch_size,
                       int num_channels) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= num_channels)
    return;

  float grad_gamma = 0.0f;
  float grad_beta = 0.0f;

  int N = batch_size * H * W;

  for(int n = 0; n < batch_size; n++) {
    for(int h = 0; h < H; h++) {
      for(int w = 0; w < W; w++) {
        int idx = n * num_channels * H * W + c * H * W + h * W + w;
        float x_hat =
            (inputs[idx] - BN_batch_stats[2 * c]) / BN_batch_stats[2 * c + 1];
        float g = grad_out[idx];
        grad_gamma += g * x_hat;
        grad_beta += g;
      }
    }
  }

  grad_weights[2 * c] = grad_gamma;
  grad_weights[2 * c + 1] = grad_beta;
}

__global__ void BN_backward_input(const float *grad_out,
                                  const float *inputs,
                                  const float *BN_batch_stats,
                                  const float *grad_weights, // gamma, beta
                                  float *grad_input,
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
  float gamma = grad_weights[2 * channel];
  float x_hat = (inputs[idx] - BN_batch_stats[2 * channel]) /
                BN_batch_stats[2 * channel + 1];
  float grad_out_val = grad_out[idx];

  // Standard BN backward formula:
  float grad_input_val =
      gamma / BN_batch_stats[2 * channel + 1] *
      (grad_out_val - grad_weights[2 * channel + 1] / N // grad_beta term
       - x_hat * grad_weights[2 * channel] / N          // grad_gamma term
      );

  grad_input[idx] = grad_input_val;
}

void cuda_BN_backward(float *grad_activations,
                      float *inputs,
                      float *BN_batch_stats,
                      float *weights,
                      float *grad_inputs,
                      float *grad_weights,
                      int num_channels,
                      int H,
                      int W,
                      int batch_size) {

  // Compute grad_gamma and grad_beta
  dim3 block_gamma(1024);
  dim3 grid_gamma((num_channels + 1023) / 1024);
  BN_backward_gamma_beta<<<grid_gamma, block_gamma>>>(grad_activations,
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
  BN_backward_input<<<grid_input, block_input>>>(grad_activations,
                                                 inputs,
                                                 BN_batch_stats,
                                                 grad_weights,
                                                 grad_inputs,
                                                 H,
                                                 W,
                                                 batch_size,
                                                 num_channels);

  cudaDeviceSynchronize();

  // dim3 gridDim(H * W / 1024, batch_size, num_channels);
  // if(H * W % 1024 != 0)
  //   gridDim.x++;
  // dim3 blockDim(1024, 1);

  // BN_backward<<<gridDim, blockDim>>>(grad_activations,
  //                                    inputs,
  //                                    BN_batch_stats,
  //                                    weights,
  //                                    grad_inputs,
  //                                    grad_weights,
  //                                    H,
  //                                    W,
  //                                    num_channels,
  //                                    batch_size);

  // cudaDeviceSynchronize();
}

__global__ void attention_psi_and_sigmoid(float *activations_int_1,
                                          float *weights,
                                          float *activations_int_2,
                                          int int_channels,
                                          int H_g,
                                          int W_g,
                                          int x_plus_g_channels) {
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H_g * W_g)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W_g;
  int w = spatial_idx % W_g;
  extern __shared__ float local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < int_channels; i += block_size) {
    local_weights[i] = weights[x_plus_g_channels * int_channels + i];
  }
  __syncthreads();
  int activation_index = flattened_index(batch_idx, 0, 1, h, H_g, w, W_g);
#pragma unroll
  for(int i = 0; i < int_channels; i++) {
    activations_int_2[activation_index] +=
        activations_int_1[flattened_index(
            batch_idx, i, int_channels, h, H_g, w, W_g)] *
        local_weights[i];
  }
  // Add last bias
  activations_int_2[activation_index] +=
      weights[(x_plus_g_channels + 1) * int_channels + int_channels];
  activations_int_2[activation_index] =
      sigmoid(activations_int_2[activation_index]);
}

__global__ void attention_resample(float *activations_int_2,
                                   float *activations,
                                   float *input_x,
                                   int H_x, // H of x
                                   int W_x, // W of x
                                   int num_channels) {
  int out_channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H_x * W_x)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W_x;
  int w = spatial_idx % W_x;
  int activation_ind =
      flattened_index(batch_idx, out_channel, num_channels, h, H_x, w, W_x);
  int small_h = h / 2;
  int small_w = w / 2;
  int A_h = small_h % 2 == 1 ? small_h - 1 : small_h;
  int A_w = small_w % 2 == 1 ? small_w - 1 : small_w;
  int A_h1 = min(A_h + 1, H_x / 2 - 1);
  int A_w1 = min(A_w + 1, W_x / 2 - 1);
  float A = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h, H_x / 2, A_w, W_x / 2)];
  float B = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h, H_x / 2, A_w + 1, W_x / 2)];
  float C = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w, W_x / 2)];
  float D = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w + 1, W_x / 2)];
  float w_int_1 = B * (w % 4 / 3.0f) + A * (1 - (w % 4 / 3.0f));
  float w_int_2 = D * (w % 4 / 3.0f) + C * (1 - (w % 4 / 3.0f));
  float interpolated_val =
      w_int_2 * (h % 4 / 3.0f) + w_int_1 * (1 - (h % 4 / 3.0f));

  activations[activation_ind] = input_x[activation_ind] * interpolated_val;
}

__global__ void attention_add_and_relu(float *input_x,
                                       float *input_g,
                                       float *weights,
                                       float *activations_int_1,
                                       int in_channels_x,
                                       int in_channels_g,
                                       int H_g, // H of g
                                       int W_g, // W of g
                                       int out_channels_int) {
  int out_channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H_g * W_g)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W_g;
  int w = spatial_idx % W_g;
  extern __shared__ float local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < in_channels_x + in_channels_g; i += block_size) {
    local_weights[i] = weights[i * out_channels_int + out_channel];
  }
  __syncthreads();
  int activation_index =
      flattened_index(batch_idx, out_channel, out_channels_int, h, H_g, w, W_g);
#pragma unroll
  for(int i = 0; i < in_channels_x; i++) {
    // Multiplications by two because x is twice the size of g. This is
    // effectively a 1x1 convolution with stride 2
    activations_int_1[activation_index] +=
        input_x[flattened_index(
            batch_idx, i, in_channels_x, h * 2, H_g * 2, w * 2, W_g * 2)] *
        local_weights[i];
  }
#pragma unroll
  for(int i = in_channels_x; i < in_channels_x + in_channels_g; i++) {
    activations_int_1[activation_index] +=
        input_g[flattened_index(
            batch_idx, i - in_channels_x, in_channels_g, h, H_g, w, W_g)] *
        local_weights[i];
  }
  // Add bias
  activations_int_1[activation_index] +=
      weights[(in_channels_x + in_channels_g + 1) * out_channels_int +
              out_channel];

  if(activations_int_1[activation_index] < 0) {
    activations_int_1[activation_index] = 0;
  }
}

void cuda_attention(float *activations_int_1,
                    float *activations_int_2,
                    float *activations,
                    float *weights,
                    float *input_x,
                    float *input_g,
                    int batch_size,
                    int H_x,
                    int W_x,
                    int channels_in_x,
                    int channels_in_g,
                    int int_channels) {
  int H_g = H_x / 2;
  int W_g = W_x / 2;
  int proj_image_size = H_g * W_g;
  dim3 blockDim(1024);
  dim3 gridDim(proj_image_size / 1024, batch_size, int_channels);
  if(proj_image_size % 1024 != 0) {
    gridDim.x++;
  }
  int shared_memory_size = sizeof(float) * channels_in_g + channels_in_x;
  attention_add_and_relu<<<gridDim, blockDim, shared_memory_size>>>(
      input_x,
      input_g,
      weights,
      activations_int_1,
      channels_in_x,
      channels_in_g,
      H_g,
      W_g,
      int_channels);

  cudaDeviceSynchronize();
  // now out channel is 1
  gridDim.z = 1;
  shared_memory_size = int_channels;
  attention_psi_and_sigmoid<<<gridDim, blockDim, shared_memory_size>>>(
      activations_int_1,
      weights,
      activations_int_2,
      int_channels,
      H_g,
      W_g,
      channels_in_x + channels_in_g);

  cudaDeviceSynchronize();
  proj_image_size = H_x * W_x;
  gridDim.x = proj_image_size / 1024;
  if(proj_image_size % 1024 != 0) {
    gridDim.x++;
  }
  gridDim.z = channels_in_x;
  attention_resample<<<gridDim, blockDim>>>(
      activations_int_2, activations, input_x, H_x, W_x, channels_in_x);

  cudaDeviceSynchronize();
}

__global__ void attention_resample_backward(float *grad_activations,
                                            float *activations_int_2,
                                            float *input_x,
                                            float *grad_activations_int_2,
                                            float *grad_input_x,
                                            int H_x,
                                            int W_x,
                                            int num_channels) {
  int out_channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H_x * W_x)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W_x;
  int w = spatial_idx % W_x;

  int activation_ind =
      flattened_index(batch_idx, out_channel, num_channels, h, H_x, w, W_x);

  int small_h = h / 2;
  int small_w = w / 2;
  int A_h = small_h % 2 == 1 ? small_h - 1 : small_h;
  int A_w = small_w % 2 == 1 ? small_w - 1 : small_w;

  float A = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h, H_x / 2, A_w, W_x / 2)];
  float B = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h, H_x / 2, A_w + 1, W_x / 2)];
  float C = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w, W_x / 2)];
  float D = activations_int_2[flattened_index(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w + 1, W_x / 2)];

  float w_frac = w % 4 / 3.0f;
  float h_frac = h % 4 / 3.0f;

  float w_int_1 = B * w_frac + A * (1 - w_frac);
  float w_int_2 = D * w_frac + C * (1 - w_frac);
  float interpolated_val = w_int_2 * h_frac + w_int_1 * (1 - h_frac);

  float grad_out = grad_activations[activation_ind];
  float x_val = input_x[activation_ind];

  // Gradient w.r.t. input_x
  grad_input_x[activation_ind] = grad_out * interpolated_val;

  // Gradient w.r.t. interpolated alpha (distribute to 4 corners)
  float grad_alpha = grad_out * x_val;

  atomicAdd(&grad_activations_int_2[flattened_index(
                batch_idx, 0, 1, A_h, H_x / 2, A_w, W_x / 2)],
            grad_alpha * (1 - w_frac) * (1 - h_frac));
  atomicAdd(&grad_activations_int_2[flattened_index(
                batch_idx, 0, 1, A_h, H_x / 2, A_w + 1, W_x / 2)],
            grad_alpha * w_frac * (1 - h_frac));
  atomicAdd(&grad_activations_int_2[flattened_index(
                batch_idx, 0, 1, A_h + 1, H_x / 2, A_w, W_x / 2)],
            grad_alpha * (1 - w_frac) * h_frac);
  atomicAdd(&grad_activations_int_2[flattened_index(
                batch_idx, 0, 1, A_h + 1, H_x / 2, A_w + 1, W_x / 2)],
            grad_alpha * w_frac * h_frac);
}

__global__ void
attention_psi_and_sigmoid_backward(float *grad_activations_int_2,
                                   float *activations_int_2,
                                   float *activations_int_1,
                                   float *weights,
                                   float *grad_activations_int_1,
                                   float *grad_weights,
                                   int int_channels,
                                   int H_g,
                                   int W_g,
                                   int x_plus_g_channels) {
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H_g * W_g)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W_g;
  int w = spatial_idx % W_g;

  extern __shared__ float local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < int_channels; i += block_size) {
    local_weights[i] = weights[x_plus_g_channels * int_channels + i];
  }
  __syncthreads();

  int activation_index = flattened_index(batch_idx, 0, 1, h, H_g, w, W_g);

  float alpha = activations_int_2[activation_index];
  float grad_alpha = grad_activations_int_2[activation_index];

  // Sigmoid backward: grad_z = grad_alpha * alpha * (1 - alpha)
  float grad_z = grad_alpha * alpha * (1 - alpha);

  // Gradient w.r.t. bias
  atomicAdd(
      &grad_weights[(x_plus_g_channels + 1) * int_channels + int_channels],
      grad_z);

  // Gradient w.r.t. psi weights and activations_int_1
  for(int i = 0; i < int_channels; i++) {
    float act = activations_int_1[flattened_index(
        batch_idx, i, int_channels, h, H_g, w, W_g)];

    atomicAdd(&grad_weights[x_plus_g_channels * int_channels + i],
              grad_z * act);

    grad_activations_int_1[flattened_index(
        batch_idx, i, int_channels, h, H_g, w, W_g)] =
        grad_z * local_weights[i];
  }
}

__global__ void attention_add_and_relu_backward(float *grad_activations_int_1,
                                                float *activations_int_1,
                                                float *input_x,
                                                float *input_g,
                                                float *weights,
                                                float *grad_input_x,
                                                float *grad_input_g,
                                                float *grad_weights,
                                                int in_channels_x,
                                                int in_channels_g,
                                                int H_g,
                                                int W_g,
                                                int out_channels_int) {
  int out_channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H_g * W_g)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W_g;
  int w = spatial_idx % W_g;

  extern __shared__ float local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < in_channels_x + in_channels_g; i += block_size) {
    local_weights[i] = weights[i * out_channels_int + out_channel];
  }
  __syncthreads();

  int activation_index =
      flattened_index(batch_idx, out_channel, out_channels_int, h, H_g, w, W_g);

  float grad_pre_relu = grad_activations_int_1[activation_index];

  // ReLU backward
  if(activations_int_1[activation_index] <= 0) {
    grad_pre_relu = 0;
  }

  // Gradient w.r.t. bias
  atomicAdd(
      &grad_weights[(in_channels_x + in_channels_g + 1) * out_channels_int +
                    out_channel],
      grad_pre_relu);

  // Gradient w.r.t. Wx and input_x
  for(int i = 0; i < in_channels_x; i++) {
    float x_val = input_x[flattened_index(
        batch_idx, i, in_channels_x, h * 2, H_g * 2, w * 2, W_g * 2)];

    atomicAdd(&grad_weights[i * out_channels_int + out_channel],
              grad_pre_relu * x_val);

    atomicAdd(&grad_input_x[flattened_index(
                  batch_idx, i, in_channels_x, h * 2, H_g * 2, w * 2, W_g * 2)],
              grad_pre_relu * local_weights[i]);
  }

  // Gradient w.r.t. Wg and input_g
  for(int i = 0; i < in_channels_g; i++) {
    float g_val =
        input_g[flattened_index(batch_idx, i, in_channels_g, h, H_g, w, W_g)];

    atomicAdd(
        &grad_weights[(in_channels_x + i) * out_channels_int + out_channel],
        grad_pre_relu * g_val);

    atomicAdd(&grad_input_g[flattened_index(
                  batch_idx, i, in_channels_g, h, H_g, w, W_g)],
              grad_pre_relu * local_weights[in_channels_x + i]);
  }
}

void cuda_attention_backward(float *grad_activations,
                             float *activations_int_1,
                             float *activations_int_2,
                             float *input_x,
                             float *input_g,
                             float *weights,
                             float *grad_activations_int_1,
                             float *grad_activations_int_2,
                             float *grad_input_x,
                             float *grad_input_g,
                             float *grad_weights,
                             int batch_size,
                             int H_x,
                             int W_x,
                             int channels_in_x,
                             int channels_in_g,
                             int int_channels) {
  int H_g = H_x / 2;
  int W_g = W_x / 2;

  // Backward through resample
  dim3 blockDim(1024);
  int proj_image_size = H_x * W_x;
  dim3 gridDim(proj_image_size / 1024, batch_size, channels_in_x);
  if(proj_image_size % 1024 != 0) {
    gridDim.x++;
  }

  attention_resample_backward<<<gridDim, blockDim>>>(grad_activations,
                                                     activations_int_2,
                                                     input_x,
                                                     grad_activations_int_2,
                                                     grad_input_x,
                                                     H_x,
                                                     W_x,
                                                     channels_in_x);

  cudaDeviceSynchronize();
  // Backward through sigmoid and psi
  proj_image_size = H_g * W_g;
  gridDim.x = proj_image_size / 1024;
  if(proj_image_size % 1024 != 0) {
    gridDim.x++;
  }
  gridDim.z = 1;
  int shared_memory_size = int_channels * sizeof(float);

  attention_psi_and_sigmoid_backward<<<gridDim, blockDim, shared_memory_size>>>(
      grad_activations_int_2,
      activations_int_2,
      activations_int_1,
      weights,
      grad_activations_int_1,
      grad_weights,
      int_channels,
      H_g,
      W_g,
      channels_in_x + channels_in_g);

  cudaDeviceSynchronize();
  // Backward through add and relu
  gridDim.z = int_channels;
  shared_memory_size = (channels_in_x + channels_in_g) * sizeof(float);

  attention_add_and_relu_backward<<<gridDim, blockDim, shared_memory_size>>>(
      grad_activations_int_1,
      activations_int_1,
      input_x,
      input_g,
      weights,
      grad_input_x,
      grad_input_g,
      grad_weights,
      channels_in_x,
      channels_in_g,
      H_g,
      W_g,
      int_channels);

  cudaDeviceSynchronize();
}

__global__ void dice_score_forward_kernel_safe(
    const float *mask,
    const float *prediction,
    float *loss_values, // output per-sample: loss_values[batch_idx*3 + 0/1/2]
    int H,
    int W) {
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  int N = H * W;

  // Each thread handles multiple pixels
  float inter_sum = 0.0f;
  float mask_sum = 0.0f;
  float pred_sum = 0.0f;

  for(int i = tid; i < N; i += blockDim.x) {
    int idx = batch_idx * N + i;
    float x = mask[idx];
    float y = prediction[idx];
    inter_sum += x * y;
    mask_sum += x;
    pred_sum += y;
  }

  // Reduce sums to global memory safely
  atomicAdd(&loss_values[batch_idx * 3 + 0], inter_sum);
  atomicAdd(&loss_values[batch_idx * 3 + 1], mask_sum);
  atomicAdd(&loss_values[batch_idx * 3 + 2], pred_sum);
}

float cuda_dice_score(const float *mask,
                      const float *prediction,
                      int H,
                      int W,
                      int batch_size,
                      float *loss_values_device) // preallocated [N*3]
{

  int threads = 1024;
  dim3 blockDim(threads);
  dim3 gridDim(batch_size); // one block per sample
  cudaMemset(loss_values_device, 0, batch_size * 3 * sizeof(float));

  dice_score_forward_kernel_safe<<<gridDim, blockDim>>>(
      mask, prediction, loss_values_device, H, W);
  cudaDeviceSynchronize();

  // Copy to CPU and compute mean dice across batch
  float *cpu_loss_values = new float[batch_size * 3];
  cudaMemcpy(cpu_loss_values,
             loss_values_device,
             batch_size * 3 * sizeof(float),
             cudaMemcpyDeviceToHost);

  float dice = 0.0f;
  for(int i = 0; i < batch_size; i++) {
    float inter = cpu_loss_values[i * 3 + 0];
    float mask_sum = cpu_loss_values[i * 3 + 1];
    float pred_sum = cpu_loss_values[i * 3 + 2];
    dice += 2.0f * inter / (mask_sum + pred_sum + 1e-5f);
  }
  delete[] cpu_loss_values;
  return dice / batch_size; // average dice
}

__global__ void
dice_loss_backward_kernel(const float *mask,
                          const float *prediction,
                          const float *loss_values, // per-sample [N*3]
                          float *grad_prediction,
                          int H,
                          int W) {
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  int N = H * W;

  float intersection = loss_values[batch_idx * 3 + 0];
  float sum_mask = loss_values[batch_idx * 3 + 1];
  float sum_pred = loss_values[batch_idx * 3 + 2];
  float eps = 1e-5f;
  float denom = sum_mask + sum_pred + eps;

  for(int i = tid; i < N; i += blockDim.x) {
    int idx = batch_idx * N + i;
    float x = mask[idx];
    float grad = 2.0f * (x * denom - intersection) / (denom * denom);
    grad_prediction[idx] = grad; // negative for gradient descent
  }
}

void cuda_dice_loss_backward(const float *mask,
                             const float *prediction,
                             const float *loss_values_device,
                             float *grad_prediction,
                             int H,
                             int W,
                             int N) {
  int block_size = 1024;
  dice_loss_backward_kernel<<<N, block_size>>>(
      mask, prediction, loss_values_device, grad_prediction, H, W);

  cudaDeviceSynchronize();
}
__global__ void dice_loss_backward(float *mask,
                                   float *prediction,
                                   float *loss_values,
                                   float *grad_prediction,
                                   int H,
                                   int W,
                                   int batch_size) {
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H * W)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W;
  int w = spatial_idx % W;

  int idx = flattened_index(batch_idx, 0, 1, h, H, w, W);
  float x = mask[idx];

  float intersection = loss_values[0];
  float sum_mask = loss_values[1];
  float sum_pred = loss_values[2];
  float denominator = sum_mask + sum_pred;

  float grad =
      2.0f * (x * denominator - intersection) / (denominator * denominator);
  grad_prediction[idx] = -grad;
}

void cuda_concat_backward(
    float *dLdY, float *dLdX, int H, int W, int C1, int C2, int batch_size) {
  int HxW = H * W;
  int C_tot = C1 + C2;
  for(int i = 0; i < batch_size; i++) {
    // first C1 channels
    cudaMemcpy(dLdX + i * C_tot * HxW,
               dLdY + i * C_tot * HxW,
               C1 * HxW * sizeof(float),
               cudaMemcpyDeviceToDevice);
    // next C2 channels
    cudaMemcpy(dLdX + i * C_tot * HxW + C1 * HxW,
               dLdY + i * C_tot * HxW + C1 * HxW,
               C2 * HxW * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
}

void cudaSetZero(float *arr, int num_floats) { cudaMemset(arr, 0, num_floats); }

__global__ void adam_update_kernel(float *weights,
                                   float *dLdW,
                                   float *adam_parameters,
                                   float learning_rate,
                                   int num_weights,
                                   float beta1,
                                   float beta2,
                                   float eps,
                                   float weight_decay,
                                   int t) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= num_weights)
    return;

  float g = dLdW[i];

  // Load previous moment estimates
  float m = adam_parameters[2 * i + 0];
  float v = adam_parameters[2 * i + 1];

  // Update biased estimates
  m = beta1 * m + (1.0f - beta1) * g;
  v = beta2 * v + (1.0f - beta2) * (g * g);

  // Save updated state
  adam_parameters[2 * i + 0] = m;
  adam_parameters[2 * i + 1] = v;

  // Bias correction
  float m_hat = m / (1.0f - powf(beta1, (float)t));
  float v_hat = v / (1.0f - powf(beta2, (float)t));

  // Update parameter
  weights[i] -= learning_rate *
                (weight_decay * weights[i] + (m_hat / (sqrtf(v_hat) + eps)));
}

void cuda_update_weights(float *weights,
                         float *dLdW,
                         float *adam_parameters,
                         float learning_rate,
                         int num_weights,
                         int t) {
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float weight_decay = 1e-4f;

  int block = 1024;
  int grid = (num_weights + block - 1) / block;

  adam_update_kernel<<<grid, block>>>(weights,
                                      dLdW,
                                      adam_parameters,
                                      learning_rate,
                                      num_weights,
                                      beta1,
                                      beta2,
                                      eps,
                                      weight_decay,
                                      t);

  cudaDeviceSynchronize();
}

void cudaLibFree(float *dPointer) { cudaFree(dPointer); }

__global__ void threshold_kernel(float *x, float threshold, int num_floats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= num_floats)
    return;

  x[i] = (x[i] >= threshold) ? 1.0f : 0.0f;
}

void cuda_threshold(float *x, float threshold, int num_floats) {
  int block = 1024;
  int grid = (num_floats + block - 1) / block;

  threshold_kernel<<<grid, block>>>(x, threshold, num_floats);

  cudaDeviceSynchronize();
}

float *cudaToCPU(float *dPointer, int num_floats) {
  float *cpu_pointer = new float[num_floats];
  cudaMemcpy(cpu_pointer,
             dPointer,
             num_floats * sizeof(float),
             cudaMemcpyDeviceToHost);
  return cpu_pointer;
}

float cudaValToCPU(float *dPointer, int ind) {
  float val;
  cudaError_t err =
      cudaMemcpy(&val, dPointer + ind, sizeof(float), cudaMemcpyDeviceToHost);

  std::stringstream ss;
  if(err != cudaSuccess) {
    ss << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
  }
  printf(ss.str().c_str());
  return val;
}

void cuda_gpu_reset() {
  cudaError_t err = cudaDeviceReset();

  std::stringstream ss;
  if(err != cudaSuccess) {
    ss << "cudaDeviceReset failed: " << cudaGetErrorString(err) << std::endl;
  }

  ss << "CUDA device reset." << std::endl;
  printf(ss.str().c_str());
}

void cuda_check_err() {

  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();

  std::stringstream ss;
  if(errSync != cudaSuccess || errAsync != cudaSuccess) {
    ss << "CUDA ERROR: "
       << cudaGetErrorString(errSync != cudaSuccess ? errSync : errAsync)
       << std::endl;
  }

  printf(ss.str().c_str());
}

void cuda_lib_copy_device(float *orig, int num_floats, float *copy) {
  cudaMemcpy(copy, orig, num_floats * sizeof(float), cudaMemcpyDeviceToDevice);
}

void cuda_lib_copy_to_device(float *cpu, float *gpu, int num_floats) {
  cudaMemcpy(gpu, cpu, num_floats * sizeof(float), cudaMemcpyHostToDevice);
}
__global__ void add_arrs_to_b(float *a, float *b, int num_floats) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < num_floats)
    b[i] += a[i];
}
void cuda_add_nums(float *a, float *b, int num_floats) {
  add_arrs_to_b<<<(num_floats + 1023) / 1024, 1024>>>(a, b, num_floats);
}

bool has_nans(const float *d_pointer, int num_floats) {
  if(d_pointer == nullptr || num_floats <= 0) {
    // Should return false for empty or null arrays, though a check for
    // cudaErrorInvalidDevicePointer might be better in a production setting.
    return false;
  }

  float *h_data = new float[num_floats];
  // 2. Copy data from Device (GPU) to Host (CPU)
  cudaError_t err = cudaMemcpy(
      h_data, d_pointer, num_floats * sizeof(float), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) {
    delete[] h_data;
    return true; // Treat copy failure as an issue
  }

  // 3. Check for NaNs on the Host
  for(int i = 0; i < num_floats; ++i) {
    // The std::isnan function is portable and reliable for checking NaN.
    // It correctly handles the fact that NaN != NaN.
    if(std::isnan(h_data[i])) {
      // You can add more context here if needed, like the value
      // std::cout << "Value: " << h_data[i] << std::endl;
      delete[] h_data;
      return true;
    }
  }

  delete[] h_data;
  return false;
}
