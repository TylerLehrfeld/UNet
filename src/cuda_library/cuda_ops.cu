#include "cuda_lib.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdexcept>

const float MOMENTUM = 0.9f;
const float INV_MOMENTUM = 1 - MOMENTUM;

__global__ void
XAVIER_or_HE_initialize(float *weights, int sqrt_N, int num_weights) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= num_weights)
    return;
  curandState state;
  unsigned long seed = 0;
  curand_init(seed, idx, 0, &state); // idx makes streams unique
  weights[idx] = curand_normal(&state) * 1.41421356237 / sqrt_N;
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
        dPointer, sqrt(N), num_weights);
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

__global__ void convolution(const float *weights,
                            const float *inputs,
                            float *activations,
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
  for(int i = threadIdx.y * blockDim.x + threadIdx.x;
      i < channels_in * kernel_size * kernel_size;
      i += block_size) {
    int c = i / (kern_squared);
    int k = i % kern_squared;
    local_kernels[c * kern_squared + k] =
        weights[flattened_index(c,
                                out_channel,
                                channels_out,
                                k / kernel_size,
                                kernel_size,
                                k % kernel_size,
                                kernel_size)];
  }
  __syncthreads();
  if(spatial_idx >= W_out * H_out)
    return;
  float sum = 0;
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
          float weight = local_kernels[c * kernel_size * kernel_size + k];
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

void convolve(const float *weights,
              const float *inputs,
              float *activations,
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

__global__ void max_pool(const float *inputs,
                         float *activations,
                         int stride,
                         int kernel_size,
                         int in_height,
                         int in_width) {
  int batch = blockIdx.z;
  int channel = threadIdx.z;
  int num_channels = blockDim.z;
  int h_out = blockIdx.x * blockDim.x + threadIdx.x;
  int w_out = blockIdx.y * blockDim.y + threadIdx.y;
  int h_in = h_out * stride;
  int w_in = w_out * stride;
  float max = INFINITY;
#pragma unroll
  for(int i = 0; i < kernel_size; i++) {
#pragma unroll
    for(int j = 0; j < kernel_size; j++) {
      int h_in_loop = h_in + i;
      int w_in_loop = w_in + j;
      if(h_in_loop < in_height && w_in_loop < in_width) {
        float val = inputs[flattened_index(batch,
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

void cuda_max_pool(const float *inputs,
                   float *activations,
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
  dim3 blockDim(1, 1024 / in_channels, in_channels);
  dim3 gridDim(in_height / stride / blockDim.x,
               in_width / stride / blockDim.y,
               batch_size);
  max_pool<<<gridDim, blockDim>>>(
      inputs, activations, stride, kernel_size, in_height, in_width);
}

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

  int shared_memory_size = scale * scale * num_out_channels * sizeof(float);
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
}

__global__ void relu(float *activations, int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations && activations[idx] < 0) {
    activations[idx] = 0;
  }
}

void cuda_relu(float *activations, int num_activations) {
  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  }
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  relu<<<gridDim, blockDim>>>(activations, num_activations);
}

__global__ void BN_inference(float *activations,
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
  activations[idx] = weight * ((activations[idx] - stats[out_channel * 2]) /
                               stats[out_channel * 2 + 1]) +
                     bias;
};

void cuda_BN_inference(float *activations,
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
      activations, weights, BN_stats, H, W, num_channels);
}

__global__ void BN_train_stats(
    float *activations, float *BN_batch_stats, int num_channels, int H, int W) {
  int channel = blockIdx.z;
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H * W)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W;
  int w = spatial_idx % W;
  int val = activations[flattened_index(
      batch_idx, channel, num_channels, h, H, w, W)];

  atomicAdd(&BN_batch_stats[channel * 2], val);
  atomicAdd(&BN_batch_stats[channel * 2 + 1], val * val);
}

__global__ void BN_train_normalize(float *activations,
                                   float *BN_batch_stats,
                                   float *BN_stats,
                                   float *weights,
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
  int mean = BN_batch_stats[channel * 2] / (H * W * batch_size);
  int var =
      (BN_batch_stats[channel * 2 + 1] - mean * mean) / (H * W * batch_size);
  int weight = weights[channel * 2];
  int bias = weights[channel * 2 + 1];
  activations[flattened_index(batch_idx, channel, num_channels, h, H, w, W)] =
      weight *
          (activations[flattened_index(
               batch_idx, channel, num_channels, h, H, w, W)] -
           mean) /
          sqrt(var) +
      bias;
  if(spatial_idx == 0 && batch_idx == 0) {
    BN_stats[channel * 2] =
        BN_stats[channel * 2] * MOMENTUM + INV_MOMENTUM * mean;
    BN_stats[channel * 2 + 1] =
        BN_stats[channel * 2 + 1] * MOMENTUM + INV_MOMENTUM * var;
  }
}

void cuda_BN_train(float *activations,
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
      activations, BN_batch_stats, num_channels, H, W);

  BN_train_normalize<<<gridDim, blockDim>>>(activations,
                                            BN_batch_stats,
                                            BN_stats,
                                            weights,
                                            H,
                                            W,
                                            num_channels,
                                            batch_size);
}

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void attention_psi_and_sigmoid(float *activations_int_1,
                                          float *weights,
                                          float *activations_int_2,
                                          int int_channels,
                                          int H_g,
                                          int W_g,
                                          int x_plus_g_channels) {
  int out_channel = blockIdx.z;
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
      sigmoid(activations_int_1[activation_index]);
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
  int shared_memory_size = channels_in_g + channels_in_x;
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
  proj_image_size = H_x * W_x;
  gridDim.x = proj_image_size / 1024;
  if(proj_image_size % 1024 != 0) {
    gridDim.x++;
  }
  gridDim.z = channels_in_x;
  attention_resample<<<gridDim, blockDim>>>(
      activations_int_2, activations, input_x, H_x, W_x, channels_in_x);
}
__global__ void dice_loss(float *mask,
                          float *prediction,
                          float *X_and_Y,
                          float *X,
                          float *Y,
                          int H,
                          int W,
                          int batch_size) {
  int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(spatial_idx >= H * W)
    return;
  int batch_idx = blockIdx.y;
  int h = spatial_idx / W;
  int w = spatial_idx % W;
  float x = mask[flattened_index(batch_idx, 0, 1, h, H, w, W)];
  float y = prediction[flattened_index(batch_idx, 0, 1, h, H, w, W)];
  atomicAdd(X_and_Y, x * y);
  atomicAdd(X, x);
  atomicAdd(Y, y);
}

float cuda_dice_loss(float *mask,
                     float *prediction,
                     int H,
                     int W,
                     int batch_size,
                     float *loss_val) {
  int proj_image_size = H * W;
  dim3 blockDim(1024);
  dim3 gridDim(proj_image_size / 1024, batch_size);
  if(proj_image_size % 1024 != 0) {
    gridDim.x++;
  }
  float *x_and_y;
  float *x = 0;
  float *y = 0;
  dice_loss<<<gridDim, blockDim>>>(
      mask, prediction, &x_and_y, &x, &y, H, W, batch_size);
  return 2 * x_and_y /
}
