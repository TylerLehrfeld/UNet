#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_max_pool(float *input, int batch_size) {
  cuda_max_pool(input,
                activations,
                batch_size,
                stride,
                kernel_size,
                in_channels,
                in_height,
                in_width);
}

inline void
Layer::backward_max_pool(float *dLdY, float *inputs, int batch_size) {
  cuda_max_pool_backward(dLdY,
                         inputs,
                         activations,
                         dLdX,
                         batch_size,
                         stride,
                         kernel_size,
                         in_channels,
                         in_height,
                         in_width);
}
