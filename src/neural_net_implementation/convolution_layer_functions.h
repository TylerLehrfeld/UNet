#include "cuda_lib.h"
#include "layer.h"
#include <stdexcept>

inline void Layer::forward_convolution(float *input, int batch_size, bool use_relu_activation) {
  // we access input as input[batch_num][channel][h][w]
  // similarly we calculate output/activations as
  // activations[batch_num][channel][h][w] Both arrays are flattened
  if(use_relu_activation) {
  convolve(parameters, input, activations,
           batch_size * out_channels * out_height * out_width, batch_size,
           kernel_size, in_channels, out_channels, out_height, out_width,
           in_height, in_width, padding, stride);
  } else {
    std::runtime_error("non-relu activated convolution not implemented");
  }
}
