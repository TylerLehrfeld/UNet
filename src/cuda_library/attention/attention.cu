#include "../cuda_lib.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

template <typename T>
__global__ void
attentionPsiAndSigmoidKernel(const T *__restrict__ activations_int_1,
                             const T *__restrict__ weights,
                             T *__restrict__ activations_int_2,
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
  extern __shared__ T local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < int_channels; i += block_size) {
    local_weights[i] = weights[x_plus_g_channels * int_channels + i];
  }
  __syncthreads();
  int activation_index = flattenIndex(batch_idx, 0, 1, h, H_g, w, W_g);
#pragma unroll
  for(int i = 0; i < int_channels; i++) {
    activations_int_2[activation_index] +=
        activations_int_1[flattenIndex(
            batch_idx, i, int_channels, h, H_g, w, W_g)] *
        local_weights[i];
  }
  // Add last bias
  activations_int_2[activation_index] +=
      weights[(x_plus_g_channels + 1) * int_channels + int_channels];
  activations_int_2[activation_index] =
      sigmoid(activations_int_2[activation_index]);
}

template <typename T>
__global__ void attentionResampleKernel(const T *__restrict__ activations_int_2,
                                        T *__restrict__ activations,
                                        const T *__restrict__ input_x,
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
      flattenIndex(batch_idx, out_channel, num_channels, h, H_x, w, W_x);
  int small_h = h / 2;
  int small_w = w / 2;
  int A_h = small_h % 2 == 1 ? small_h - 1 : small_h;
  int A_w = small_w % 2 == 1 ? small_w - 1 : small_w;
  int A_h1 = min(A_h + 1, H_x / 2 - 1);
  int A_w1 = min(A_w + 1, W_x / 2 - 1);
  T A = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h, H_x / 2, A_w, W_x / 2)];
  T B = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h, H_x / 2, A_w + 1, W_x / 2)];
  T C = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w, W_x / 2)];
  T D = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w + 1, W_x / 2)];
  T w_int_1 = B * (w % 4 / 3.0f) + A * (1 - (w % 4 / 3.0f));
  T w_int_2 = D * (w % 4 / 3.0f) + C * (1 - (w % 4 / 3.0f));
  T interpolated_val =
      w_int_2 * (h % 4 / 3.0f) + w_int_1 * (1 - (h % 4 / 3.0f));

  activations[activation_ind] = input_x[activation_ind] * interpolated_val;
}

template <typename T>
__global__ void attentionAddAndReluKernel(const T *__restrict__ input_x,
                                          const T *__restrict__ input_g,
                                          const T *__restrict__ weights,
                                          T *__restrict__ activations_int_1,
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
  extern __shared__ T local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < in_channels_x + in_channels_g; i += block_size) {
    local_weights[i] = weights[i * out_channels_int + out_channel];
  }
  __syncthreads();
  int activation_index =
      flattenIndex(batch_idx, out_channel, out_channels_int, h, H_g, w, W_g);
#pragma unroll
  for(int i = 0; i < in_channels_x; i++) {
    // Multiplications by two because x is twice the size of g. This is
    // effectively a 1x1 convolution with stride 2
    activations_int_1[activation_index] +=
        input_x[flattenIndex(
            batch_idx, i, in_channels_x, h * 2, H_g * 2, w * 2, W_g * 2)] *
        local_weights[i];
  }
#pragma unroll
  for(int i = in_channels_x; i < in_channels_x + in_channels_g; i++) {
    activations_int_1[activation_index] +=
        input_g[flattenIndex(
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

template <typename T>
void cudaAttention(T *__restrict__ activations_int_1,
                   T *__restrict__ activations_int_2,
                   T *__restrict__ activations,
                   const T *__restrict__ weights,
                   const T *__restrict__ input_x,
                   const T *__restrict__ input_g,
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
  int shared_memory_size = sizeof(T) * channels_in_g + channels_in_x;
  attentionAddAndReluKernel<<<gridDim, blockDim, shared_memory_size>>>(
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
  attentionPsiAndSigmoidKernel<<<gridDim, blockDim, shared_memory_size>>>(
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
  attentionResampleKernel<<<gridDim, blockDim>>>(
      activations_int_2, activations, input_x, H_x, W_x, channels_in_x);

  cudaDeviceSynchronize();
}
template void cudaAttention<float>(float *__restrict__ activations_int_1,
                                   float *__restrict__ activations_int_2,
                                   float *__restrict__ activations,
                                   const float *__restrict__ weights,
                                   const float *__restrict__ input_x,
                                   const float *__restrict__ input_g,
                                   int batch_size,
                                   int H_x,
                                   int W_x,
                                   int channels_in_x,
                                   int channels_in_g,
                                   int int_channels);

template <typename T>
__global__ void
attentionResampleBackwardKernel(const T *__restrict__ grad_activations,
                                const T *__restrict__ activations_int_2,
                                const T *__restrict__ input_x,
                                T *__restrict__ grad_activations_int_2,
                                T *__restrict__ grad_input_x,
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
      flattenIndex(batch_idx, out_channel, num_channels, h, H_x, w, W_x);

  int small_h = h / 2;
  int small_w = w / 2;
  int A_h = small_h % 2 == 1 ? small_h - 1 : small_h;
  int A_w = small_w % 2 == 1 ? small_w - 1 : small_w;

  T A = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h, H_x / 2, A_w, W_x / 2)];
  T B = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h, H_x / 2, A_w + 1, W_x / 2)];
  T C = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w, W_x / 2)];
  T D = activations_int_2[flattenIndex(
      batch_idx, 0, 1, A_h + 1, H_x / 2, A_w + 1, W_x / 2)];

  T w_frac = w % 4 / 3.0f;
  T h_frac = h % 4 / 3.0f;

  T w_int_1 = B * w_frac + A * (1 - w_frac);
  T w_int_2 = D * w_frac + C * (1 - w_frac);
  T interpolated_val = w_int_2 * h_frac + w_int_1 * (1 - h_frac);

  T grad_out = grad_activations[activation_ind];
  T x_val = input_x[activation_ind];

  // Gradient w.r.t. input_x
  grad_input_x[activation_ind] = grad_out * interpolated_val;

  // Gradient w.r.t. interpolated alpha (distribute to 4 corners)
  T grad_alpha = grad_out * x_val;

  atomicAdd(&grad_activations_int_2[flattenIndex(
                batch_idx, 0, 1, A_h, H_x / 2, A_w, W_x / 2)],
            grad_alpha * (1 - w_frac) * (1 - h_frac));
  atomicAdd(&grad_activations_int_2[flattenIndex(
                batch_idx, 0, 1, A_h, H_x / 2, A_w + 1, W_x / 2)],
            grad_alpha * w_frac * (1 - h_frac));
  atomicAdd(&grad_activations_int_2[flattenIndex(
                batch_idx, 0, 1, A_h + 1, H_x / 2, A_w, W_x / 2)],
            grad_alpha * (1 - w_frac) * h_frac);
  atomicAdd(&grad_activations_int_2[flattenIndex(
                batch_idx, 0, 1, A_h + 1, H_x / 2, A_w + 1, W_x / 2)],
            grad_alpha * w_frac * h_frac);
}

template <typename T>
__global__ void attentionPsiAndSigmoidBackwardKernel(
    const T *__restrict__ grad_activations_int_2,
    const T *__restrict__ activations_int_2,
    const T *__restrict__ activations_int_1,
    const T *__restrict__ weights,
    T *__restrict__ grad_activations_int_1,
    T *__restrict__ grad_weights,
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

  extern __shared__ T local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < int_channels; i += block_size) {
    local_weights[i] = weights[x_plus_g_channels * int_channels + i];
  }
  __syncthreads();

  int activation_index = flattenIndex(batch_idx, 0, 1, h, H_g, w, W_g);

  T alpha = activations_int_2[activation_index];
  T grad_alpha = grad_activations_int_2[activation_index];

  // Sigmoid backward: grad_z = grad_alpha * alpha * (1 - alpha)
  T grad_z = grad_alpha * alpha * (1 - alpha);

  // Gradient w.r.t. bias
  atomicAdd(
      &grad_weights[(x_plus_g_channels + 1) * int_channels + int_channels],
      grad_z);

  // Gradient w.r.t. psi weights and activations_int_1
  for(int i = 0; i < int_channels; i++) {
    T act = activations_int_1[flattenIndex(
        batch_idx, i, int_channels, h, H_g, w, W_g)];

    atomicAdd(&grad_weights[x_plus_g_channels * int_channels + i],
              grad_z * act);

    grad_activations_int_1[flattenIndex(
        batch_idx, i, int_channels, h, H_g, w, W_g)] =
        grad_z * local_weights[i];
  }
}

template <typename T>
__global__ void
attentionAddAndReluBackwardKernel(const T *__restrict__ grad_activations_int_1,
                                  const T *__restrict__ activations_int_1,
                                  const T *__restrict__ input_x,
                                  const T *__restrict__ input_g,
                                  const T *__restrict__ weights,
                                  T *__restrict__ grad_input_x,
                                  T *__restrict__ grad_input_g,
                                  T *__restrict__ grad_weights,
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

  extern __shared__ T local_weights[];
  int block_size = blockDim.x;
  for(int i = threadIdx.x; i < in_channels_x + in_channels_g; i += block_size) {
    local_weights[i] = weights[i * out_channels_int + out_channel];
  }
  __syncthreads();

  int activation_index =
      flattenIndex(batch_idx, out_channel, out_channels_int, h, H_g, w, W_g);

  T grad_pre_relu = grad_activations_int_1[activation_index];

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
    T x_val = input_x[flattenIndex(
        batch_idx, i, in_channels_x, h * 2, H_g * 2, w * 2, W_g * 2)];

    atomicAdd(&grad_weights[i * out_channels_int + out_channel],
              grad_pre_relu * x_val);

    atomicAdd(&grad_input_x[flattenIndex(
                  batch_idx, i, in_channels_x, h * 2, H_g * 2, w * 2, W_g * 2)],
              grad_pre_relu * local_weights[i]);
  }

  // Gradient w.r.t. Wg and input_g
  for(int i = 0; i < in_channels_g; i++) {
    T g_val =
        input_g[flattenIndex(batch_idx, i, in_channels_g, h, H_g, w, W_g)];

    atomicAdd(
        &grad_weights[(in_channels_x + i) * out_channels_int + out_channel],
        grad_pre_relu * g_val);

    atomicAdd(&grad_input_g[flattenIndex(
                  batch_idx, i, in_channels_g, h, H_g, w, W_g)],
              grad_pre_relu * local_weights[in_channels_x + i]);
  }
}

template <typename T>
void cudaAttentionBackward(const T *__restrict__ grad_activations,
                           const T *__restrict__ activations_int_1,
                           const T *__restrict__ activations_int_2,
                           const T *__restrict__ input_x,
                           const T *__restrict__ input_g,
                           const T *__restrict__ weights,
                           T *__restrict__ grad_activations_int_1,
                           T *__restrict__ grad_activations_int_2,
                           T *__restrict__ grad_input_x,
                           T *__restrict__ grad_input_g,
                           T *__restrict__ grad_weights,
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

  attentionResampleBackwardKernel<<<gridDim, blockDim>>>(grad_activations,
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
  int shared_memory_size = int_channels * sizeof(T);

  attentionPsiAndSigmoidBackwardKernel<<<gridDim,
                                         blockDim,
                                         shared_memory_size>>>(
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
  shared_memory_size = (channels_in_x + channels_in_g) * sizeof(T);

  attentionAddAndReluBackwardKernel<<<gridDim, blockDim, shared_memory_size>>>(
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

template void
cudaAttentionBackward<float>(const float *__restrict__ grad_activations,
                             const float *__restrict__ activations_int_1,
                             const float *__restrict__ activations_int_2,
                             const float *__restrict__ input_x,
                             const float *__restrict__ input_g,
                             const float *__restrict__ weights,
                             float *__restrict__ grad_activations_int_1,
                             float *__restrict__ grad_activations_int_2,
                             float *__restrict__ grad_input_x,
                             float *__restrict__ grad_input_g,
                             float *__restrict__ grad_weights,
                             int batch_size,
                             int H_x,
                             int W_x,
                             int channels_in_x,
                             int channels_in_g,
                             int int_channels);
