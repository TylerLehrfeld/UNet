#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_convolution(float *input, int batch_size) {
  // we access input as input[batch_num][channel][h][w]
  // similarly we calculate output/activations as
  // activations[batch_num][channel][h][w] Both arrays are flattened
  convolve(parameters, input, activations,
           batch_size * out_channels * out_height * out_width, batch_size,
           kernel_size, in_channels, out_channels, out_height, out_width,
           in_height, in_width, padding, stride);
}
