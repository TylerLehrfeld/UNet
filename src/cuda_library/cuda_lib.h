#ifndef UNET_CUDA_LIB
#define UNET_CUDA_LIB

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

enum Initiation_type {
  XAVIER,
  ZERO,
  HE,
  BN, // for batch norm identity initiation
};

__host__ __device__ inline int flattenIndex(
    int batch_num, int channel, int num_channels, int h, int H, int w, int W) {
  return batch_num * (num_channels * H * W) + channel * (H * W) + h * W + w;
}

template <typename T> __device__ inline float sigmoid(T x) {
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
}

const float kMomentum = 0.9f;
const float kInverse_momentum = 1 - kMomentum;

template <typename T>
void cudaAddBtoA(T *__restrict__ a, const T *__restrict__ b, int num_floats);

template <typename T>
void cudaDifferenceLossBackward(const T *__restrict__ mask,
                                const T *__restrict__ y,
                                T *__restrict__ dLdY,
                                int total_size);

template <typename T>
T *cudaGetPointer(int num_floats,
                  Initiation_type i_type = ZERO,
                  int N = 0,
                  int num_weights = 0);

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
                  int stride);

template <typename T>
void cudaConcat(const T *__restrict__ activations_1,
                const T *__restrict__ activations_2,
                T *__restrict__ activations,
                int batch_size,
                int HxW,
                int C1,
                int C2);

template <typename T>
void cudaMaxPool(const T *__restrict__ inputs,
                 T *__restrict__ activations,
                 int batch_size,
                 int stride,
                 int kernel_size,
                 int in_channels,
                 int in_height,
                 int in_width);

template <typename T>
void cudaUpsample(const T *__restrict__ inputs,
                  const T *__restrict__ weights,
                  T *__restrict__ activations,
                  int scale,
                  int num_in_channels,
                  int num_out_channels,
                  int H_in,
                  int W_in,
                  int batch_size);

template <typename T>
void cudaRelu(const T *__restrict__ inputs,
              T *__restrict__ activations,
              int num_activations);

template <typename T>
void cudaSigmoid(T *__restrict__ inputs,
                 T *__restrict__ activations,
                 int num_activations);

template <typename T>
void cudaBatchNormTrain(const T *__restrict__ inputs,
                        T *__restrict__ activations,
                        const T *__restrict__ weights,
                        T *__restrict__ BN_batch_stats,
                        T *__restrict__ BN_running_stats,
                        int num_channels,
                        int H,
                        int W,
                        int batch_size);

template <typename T>
void cudaBatchNormInference(const T *__restrict__ inputs,
                            T *__restrict__ activations,
                            const T *__restrict__ weights,
                            const T *__restrict__ BN_stats,
                            int H,
                            int W,
                            int num_channels);

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
                   int int_channels);

template <typename T>
float cudaDiceScore(const T *__restrict__ mask,
                    const T *__restrict__ prediction,
                    int H,
                    int W,
                    int batch_size,
                    T *__restrict__ loss_values);

template <typename T>
void cudaDiceLossBackward(const T *__restrict__ mask,
                          const T *__restrict__ prediction,
                          const T *__restrict__ loss_values,
                          T *__restrict__ dLdY,
                          int H,
                          int W,
                          int batch_size);

template <typename T>
void cudaReluBackward(const T *__restrict__ grad_activations,
                      const T *__restrict__ inputs,
                      T *__restrict__ grad_inputs,
                      int num_activations);

template <typename T>
void cudaSigmoidBackward(const T *__restrict__ grad_activations,
                         const T *__restrict__ activations,
                         T *__restrict__ grad_inputs,
                         int num_activations);
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
                          int stride);

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
                           int batch_size);

template <typename T>
void cudaConcatBackward(T *__restrict__ dLdY,
                        T *__restrict__ dLdX,
                        int H,
                        int W,
                        int C1,
                        int C2,
                        int batch_size);

template <typename T> void cudaSetZero(T *arr, int num_floats);

template <typename T>
void cudaMaxPoolBackward(const T *__restrict__ grad_activations,
                         const T *__restrict__ inputs,
                         const T *__restrict__ activations,
                         T *__restrict__ grad_inputs,
                         int batch_size,
                         int stride,
                         int in_channels,
                         int in_height,
                         int in_width);

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
                          int batch_size);

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
                           int int_channels);

template <typename T>
void cudaUpdateWeights(T *__restrict__ weights,
                       const T *__restrict__ dLdW,
                       T *__restrict__ adam_parameters,
                       float learning_rate,
                       int num_weights,
                       int t);
template <typename T>
void cudaUpdateWeights(T *__restrict__ weights,
                       const T *__restrict__ dLdW,
                       float learning_rate,
                       int num_weights);

template <typename T> void cudaLibFree(T *dPointer);
template <typename T> void cudaThreshold(T *x, T threshold, int num_elements);
template <typename T> T *cudaToCPU(T *dPointer, int num_elements);
void cudaGPUReset();
void cudaCheckErr();
template <typename T> T cudaValToCPU(T *dPointer, int ind);
template <typename T>
void cudaLibCopyDeviceToDevice(const T *dOrig, int num_elements, T *dCopy);
template <typename T>
void cudaLibCopyToDevice(const T *cpu, T *gpu, int num_elements);
template <typename T>
void cudaLibCopyToHost(T *cpu, const T *gpu, int num_elements);
template <typename T> bool cudaHasNans(const T *d_pointer, int num_elements);
#endif
