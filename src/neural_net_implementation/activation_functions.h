#include "cuda_lib.h"

#ifndef ACTIVATIONS_UNET_FUNCTIONS
#define ACTIVATIONS_UNET_FUNCTIONS

inline void
forward_relu(float *inputs, float *activations, int num_activations) {
  cudaRelu(inputs, activations, num_activations);
}

inline void
forward_sigmoid(float *inputs, float *activations, int num_activations) {
  cudaSigmoid(inputs, activations, num_activations);
}

#endif // !ACTIVATIONS_UNET_FUNCTIONS
