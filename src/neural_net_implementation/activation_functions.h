#include "cuda_lib.h"

#ifndef ACTIVATIONS_UNET_FUNCTIONS
#define ACTIVATIONS_UNET_FUNCTIONS

inline void forward_relu(float *activations, int num_activations) {
  cuda_relu(activations, num_activations);
}

#endif // !ACTIVATIONS_UNET_FUNCTIONS
