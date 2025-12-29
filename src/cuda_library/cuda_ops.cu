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

template <typename T>
__global__ void XAVIER_or_HE_initialize(T *weights, T sqrt_N, int num_weights) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= num_weights)
    return;
  curandState state;
  unsigned long seed = 0;
  curand_init(seed, idx, 0, &state); // idx makes streams unique
  float norm_samp = curand_normal(&state);
  weights[idx] = norm_samp * static_cast<T>(1.41421356237) / sqrt_N;
}

template <typename T>
__global__ void BatchNormInitializeKernel(T *weights, int num_weights) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= num_weights)
    return;
  weights[idx] = static_cast<T>(idx % 2 == 0);
}

template <typename T>
__host__ T *
cudaGetPointer(int num_T, Initiation_type i_type, int N, int num_weights) {
  T *dPointer;
  cudaMalloc(&dPointer, num_T * sizeof(T));
  cudaMemset(dPointer, 0, num_T * sizeof(T));
  // TODO: Consider making block size variable to num_weights
  int block_size = 128;
  dim3 blockDim(block_size);
  dim3 gridDim((num_weights + blockDim.x - 1) / blockDim.x);

  if(i_type == XAVIER || i_type == HE) {
    if(N == 0) {
      throw std::runtime_error("Can't pass N=0 to cudaGetPointer when "
                               "initialzing with HE or XAVIER");
    }
    XAVIER_or_HE_initialize<<<gridDim, blockDim>>>(
        dPointer, sqrtf(N), num_weights);

  } else if(i_type == BN) {
    BatchNormInitializeKernel<<<gridDim, blockDim>>>(dPointer, num_weights);
  }

  cudaDeviceSynchronize();
  return dPointer;
}

template float *cudaGetPointer<float>(int num_T,
                                      Initiation_type i_type,
                                      int N,
                                      int num_weights);

template <typename T>
void cudaConcat(const T *__restrict__ activations_1,
                const T *__restrict__ activations_2,
                T *__restrict__ activations,
                int batch_size,
                int HxW,
                int C1,
                int C2) {

  int C_tot = C1 + C2;
  for(int i = 0; i < batch_size; i++) {
    cudaMemcpy(activations + i * ((C_tot)*HxW),
               activations_1 + i * C1 * HxW,
               C1 * HxW * sizeof(T),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(activations + i * (C_tot)*HxW + C1 * HxW,
               activations_2 + i * C2 * HxW,
               C2 * HxW * sizeof(T),
               cudaMemcpyDeviceToDevice);
  }
}
template void cudaConcat<float>(const float *__restrict__ activations_1,
                                const float *__restrict__ activations_2,
                                float *__restrict__ activations,
                                int batch_size,
                                int HxW,
                                int C1,
                                int C2);

template <typename T>
__global__ void reluKernel(const T *__restrict__ inputs,
                           T *__restrict__ activations,
                           int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    float val = inputs[idx];
    activations[idx] = max(static_cast<T>(0.0), inputs[idx]);
  }
}

template <typename T>
void cudaRelu(const T *__restrict__ inputs,
              T *__restrict__ activations,
              int num_activations) {

  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  }
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  reluKernel<<<gridDim, blockDim>>>(inputs, activations, num_activations);

  cudaDeviceSynchronize();
}

template void cudaRelu<float>(const float *__restrict__ inputs,
                              float *__restrict__ activations,
                              int num_activations);

template <typename T>
__global__ void sigmoidKernel(T *inputs, T *activations, int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    activations[idx] = sigmoid(inputs[idx]);
  }
}
template <typename T>
void cudaSigmoid(T *__restrict__ inputs,
                 T *__restrict__ activations,
                 int num_activations) {

  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  };
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  sigmoidKernel<<<gridDim, blockDim>>>(inputs, activations, num_activations);

  cudaDeviceSynchronize();
}
template void cudaSigmoid<float>(float *__restrict__ inputs,
                                 float *__restrict__ activations,
                                 int num_activations);

template <typename T>
__global__ void sigmoidBackwardKernel(const T *__restrict__ grad_activations,
                                      const T *__restrict__ activations,
                                      T *__restrict__ grad_inputs,
                                      int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    float sig = activations[idx];
    grad_inputs[idx] = grad_activations[idx] * sig * (1.0f - sig);
  }
}

template <typename T>
void cudaSigmoidBackward(const T *__restrict__ grad_activations,
                         const T *__restrict__ activations,
                         T *__restrict__ grad_inputs,
                         int num_activations) {
  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  }
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  sigmoidBackwardKernel<<<gridDim, blockDim>>>(
      grad_activations, activations, grad_inputs, num_activations);

  cudaDeviceSynchronize();
}

template void
cudaSigmoidBackward<float>(const float *__restrict__ grad_activations,
                           const float *__restrict__ activations,
                           float *__restrict__ grad_inputs,
                           int num_activations);

template <typename T>
__global__ void reluBackwardKernel(const T *__restrict__ grad_activations,
                                   const T *__restrict__ inputs,
                                   T *__restrict__ grad_inputs,
                                   int num_activations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x +
            blockIdx.y * blockDim.x * gridDim.x;
  if(idx < num_activations) {
    grad_inputs[idx] = (inputs[idx] > 0) ? grad_activations[idx] : 0.0f;
  }
}

template <typename T>
void cudaReluBackward(const T *__restrict__ grad_activations,
                      const T *__restrict__ inputs,
                      T *__restrict__ grad_inputs,
                      int num_activations) {

  int num_blocks = num_activations / 1024;
  if(num_activations % 1024 != 0) {
    num_blocks++;
  }
  dim3 gridDim(num_blocks, 1);
  dim3 blockDim(1024, 1);
  reluBackwardKernel<<<gridDim, blockDim>>>(
      grad_activations, inputs, grad_inputs, num_activations);

  cudaDeviceSynchronize();
}
template void
cudaReluBackward<float>(const float *__restrict__ grad_activations,
                        const float *__restrict__ inputs,
                        float *__restrict__ grad_inputs,
                        int num_activations);

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

template <typename T>
float cudaDiceScore(const T *__restrict__ mask,
                    const T *__restrict__ prediction,
                    int H,
                    int W,
                    int batch_size,
                    T *__restrict__ loss_values) {

  int threads = 1024;
  dim3 blockDim(threads);
  dim3 gridDim(batch_size); // one block per sample
  cudaMemset(loss_values, 0, batch_size * 3 * sizeof(float));

  dice_score_forward_kernel_safe<<<gridDim, blockDim>>>(
      mask, prediction, loss_values, H, W);
  cudaDeviceSynchronize();

  // Copy to CPU and compute mean dice across batch
  float *cpu_loss_values = new float[batch_size * 3];
  cudaMemcpy(cpu_loss_values,
             loss_values,
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

template float
cudaDiceScore(const float *__restrict__ mask,
              const float *__restrict__ prediction,
              int H,
              int W,
              int batch_size,
              float *__restrict__ loss_values_device); // preallocated [N*3]

template <typename T>
__global__ void
diceLossBackwardKernel(const T *__restrict__ mask,
                       const T *__restrict__ prediction,
                       const T *__restrict__ loss_values, // per-sample [N*3]
                       T *__restrict__ grad_prediction,
                       int H,
                       int W) {
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  int N = H * W;

  T intersection = loss_values[batch_idx * 3 + 0];
  T sum_mask = loss_values[batch_idx * 3 + 1];
  T sum_pred = loss_values[batch_idx * 3 + 2];
  T eps = 1e-5f;
  T denom = sum_mask + sum_pred + eps;

  for(int i = tid; i < N; i += blockDim.x) {
    int idx = batch_idx * N + i;
    T x = mask[idx];
    T grad = 2.0f * (x * denom - intersection) / (denom * denom);
    grad_prediction[idx] = grad; // negative for gradient descent
  }
}

template <typename T>
void cudaDiceLossBackward(const T *__restrict__ mask,
                          const T *__restrict__ prediction,
                          const T *__restrict__ loss_values,
                          T *__restrict__ dLdY,
                          int H,
                          int W,
                          int batch_size) {

  int block_size = 1024;

  diceLossBackwardKernel<<<batch_size, block_size>>>(
      mask, prediction, loss_values, dLdY, H, W);

  cudaDeviceSynchronize();
}

template void
cudaDiceLossBackward<float>(const float *__restrict__ mask,
                            const float *__restrict__ prediction,
                            const float *__restrict__ loss_values_device,
                            float *__restrict__ grad_prediction,
                            int H,
                            int W,
                            int N);

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

  int idx = flattenIndex(batch_idx, 0, 1, h, H, w, W);
  float x = mask[idx];

  float intersection = loss_values[0];
  float sum_mask = loss_values[1];
  float sum_pred = loss_values[2];
  float denominator = sum_mask + sum_pred;

  float grad =
      2.0f * (x * denominator - intersection) / (denominator * denominator);
  grad_prediction[idx] = -grad;
}

template <typename T>
void cudaConcatBackward(T *__restrict__ dLdY,
                        T *__restrict__ dLdX,
                        int H,
                        int W,
                        int C1,
                        int C2,
                        int batch_size) {

  int HxW = H * W;
  int C_tot = C1 + C2;
  for(int i = 0; i < batch_size; i++) {
    // first C1 channels
    cudaMemcpy(dLdX + i * C_tot * HxW,
               dLdY + i * C_tot * HxW,
               C1 * HxW * sizeof(T),
               cudaMemcpyDeviceToDevice);
    // next C2 channels
    cudaMemcpy(dLdX + i * C_tot * HxW + C1 * HxW,
               dLdY + i * C_tot * HxW + C1 * HxW,
               C2 * HxW * sizeof(T),
               cudaMemcpyDeviceToDevice);
  }
}

template void cudaConcatBackward<float>(
    float *dLdY, float *dLdX, int H, int W, int C1, int C2, int batch_size);

void cudaSetZero(float *arr, int num_floats) { cudaMemset(arr, 0, num_floats); }

template <typename T>
__global__ void adamUpdateKernel(T *__restrict__ weights,
                                 const T *__restrict__ dLdW,
                                 T *__restrict__ adam_parameters,
                                 T learning_rate,
                                 int num_weights,
                                 T beta1,
                                 T beta2,
                                 T eps,
                                 T weight_decay,
                                 int t) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= num_weights)
    return;

  T g = dLdW[i];

  // Load previous moment estimates
  T m = adam_parameters[2 * i + 0];
  T v = adam_parameters[2 * i + 1];

  // Update biased estimates
  m = beta1 * m + (1.0 - beta1) * g;
  v = beta2 * v + (1.0 - beta2) * (g * g);

  // Save updated state
  adam_parameters[2 * i + 0] = m;
  adam_parameters[2 * i + 1] = v;

  // Bias correction
  T m_hat = m / (1.0 - pow(beta1, (float)t));
  T v_hat = v / (1.0 - pow(beta2, (float)t));

  // Update parameter
  weights[i] -= learning_rate *
                (weight_decay * weights[i] + (m_hat / (sqrt(v_hat) + eps)));
}

template <typename T>
void cudaUpdateWeights(T *__restrict__ weights,
                       const T *__restrict__ dLdW,
                       T *__restrict__ adam_parameters,
                       float learning_rate,
                       int num_weights,
                       int t) {

  const T beta1 = static_cast<T>(0.9f);
  const T beta2 = static_cast<T>(0.999f);
  const T eps = static_cast<T>(1e-8f);
  const T weight_decay = static_cast<T>(1e-4f);

  int block = 1024;
  int grid = (num_weights + block - 1) / block;

  adamUpdateKernel<<<grid, block>>>(weights,
                                    dLdW,
                                    adam_parameters,
                                    static_cast<T>(learning_rate),
                                    num_weights,
                                    beta1,
                                    beta2,
                                    eps,
                                    weight_decay,
                                    t);

  cudaDeviceSynchronize();
}

template void cudaUpdateWeights<float>(float *__restrict__ weights,
                                       const float *__restrict__ dLdW,
                                       float *__restrict__ adam_parameters,
                                       float learning_rate,
                                       int num_weights,
                                       int t);

template <typename T>
__global__ void genericUpdateWeightsKernel(T *__restrict__ weights,
                                           const T *__restrict__ dLdW,
                                           T learning_rate,
                                           int num_weights) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= num_weights) {
    return;
  }
  weights[idx] -= dLdW[idx] * learning_rate;
}

template <typename T>
void cudaUpdateWeights(T *__restrict__ weights,
                       const T *__restrict__ dLdW,
                       float learning_rate,
                       int num_weights) {
  int block_size = 768;
  int num_blocks = num_weights / 768 + 1;
  genericUpdateWeightsKernel<<<num_blocks, block_size>>>(
      weights, dLdW, learning_rate, num_weights);
}

template void cudaUpdateWeights<float>(float *__restrict__ weights,
                                       const float *__restrict__ dLdW,
                                       float learning_rate,
                                       int num_weights);

template <typename T> void cudaLibFree(T *dPointer) { cudaFree(dPointer); }
template void cudaLibFree<float>(float *dPointer);

template <typename T>
__global__ void thresholdKernel(T *x, T threshold, int num_floats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= num_floats)
    return;

  x[i] = (x[i] >= threshold) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template <typename T> void cudaThreshold(T *x, T threshold, int num_floats) {
  int block = 1024;
  int grid = (num_floats + block - 1) / block;

  thresholdKernel<<<grid, block>>>(x, threshold, num_floats);

  cudaDeviceSynchronize();
}

template void cudaThreshold<float>(float *x, float threshold, int num_floats);

template <typename T> T *cudaToCPU(T *dPointer, int num_elements) {

  T *cpu_pointer = new T[num_elements];
  cudaMemcpy(
      cpu_pointer, dPointer, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
  return cpu_pointer;
}

template float *cudaToCPU(float *dPointer, int num_elements);

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

void cudaGPUReset() {
  cudaError_t err = cudaDeviceReset();

  std::stringstream ss;
  if(err != cudaSuccess) {
    ss << "cudaDeviceReset failed: " << cudaGetErrorString(err) << std::endl;
  }

  ss << "CUDA device reset." << std::endl;
  printf(ss.str().c_str());
}

void cudaCheckErr() {
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

template <typename T>
void cudaLibCopyDeviceToDevice(const T *dOrig, int num_floats, T *dCopy) {

  cudaMemcpy(
      dCopy, dOrig, num_floats * sizeof(float), cudaMemcpyDeviceToDevice);
}
template void cudaLibCopyDeviceToDevice<float>(const float *dOrig,
                                               int num_floats,
                                               float *dCopy);

template <typename T>
void cudaLibCopyToDevice(const T *cpu, T *gpu, int num_floats) {

  cudaMemcpy(gpu, cpu, num_floats * sizeof(T), cudaMemcpyHostToDevice);
}

template void
cudaLibCopyToDevice<float>(const float *cpu, float *gpu, int num_floats);

template <typename T>
__global__ void
addArrayBtoAKernel(T *__restrict__ a, const T *__restrict__ b, int num_floats) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < num_floats)
    a[i] += b[i];
}

template <typename T>
void cudaAddBtoA(T *__restrict__ a, const T *__restrict__ b, int num_floats) {
  addArrayBtoAKernel<<<(num_floats + 1023) / 1024, 1024>>>(a, b, num_floats);
}
template void cudaAddBtoA<float>(float *__restrict__ a,
                                 const float *__restrict__ b,
                                 int num_floats);

template <typename T>
__global__ void differenceLossBackwardKernel(const T *__restrict__ mask,
                                             const T *y,
                                             T *dLdY,
                                             int total_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= total_size) {
    return;
  }
  dLdY[idx] = (y[idx] - mask[idx]) / total_size;
}

template <typename T>
void cudaDifferenceLossBackward(const T *__restrict__ mask,
                                const T *__restrict__ y,
                                T *__restrict__ dLdY,
                                int total_size) {
  int block_size = 768;
  dim3 blockDim(block_size);
  dim3 gridDim(total_size / block_size);
  if(total_size % block_size != 0) {
    gridDim.x++;
  }
  differenceLossBackwardKernel<<<gridDim, blockDim>>>(
      mask, y, dLdY, total_size);
  cudaDeviceSynchronize();
}
template void cudaDifferenceLossBackward<float>(const float *__restrict__ mask,
                                                const float *__restrict__ y,
                                                float *__restrict__ dLdY,
                                                int total_size);

// TODO: Make this done on the kernel. Not really necessary for testing right
// now though
template <typename T> bool cudaHasNans(const T *d_pointer, int num_floats) {

  if(d_pointer == nullptr || num_floats <= 0) {
    return false;
  }

  T *h_data = new T[num_floats];
  cudaError_t err = cudaMemcpy(
      h_data, d_pointer, num_floats * sizeof(float), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) {
    delete[] h_data;
    return true;
  }

  for(int i = 0; i < num_floats; ++i) {
    if(std::isnan(h_data[i])) {
      delete[] h_data;
      return true;
    }
  }

  delete[] h_data;
  return false;
}

template bool cudaHasNans<float>(const float *d_pointer, int num_floats);

template <typename T>
void cudaLibCopyToHost(T *cpu, const T *gpu, int num_elements) {
  cudaMemcpy(cpu, gpu, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
}
template void
cudaLibCopyToHost<float>(float *cpu, const float *gpu, int num_elements);
