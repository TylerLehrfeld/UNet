#include "BN_functions.h"
#include "activation_functions.h"
#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_convolution(float *input,
                                       int batch_size,
                                       bool use_relu_activation,
                                       bool use_batch_norm,
                                       bool inference) {
  // we access input as input[batch_num][channel][h][w]
  // similarly we calculate output/activations as
  // activations[batch_num][channel][h][w] Both arrays are flattened
  int num_activations = batch_size * out_channels * out_height * out_width;
  convolve(parameters,
           input,
           activations,
           num_activations,
           batch_size,
           kernel_size,
           in_channels,
           out_channels,
           out_height,
           out_width,
           in_height,
           in_width,
           padding,
           stride);
  if(inference) {
    forward_batch_norm_inference(activations,
                                 BN_parameters,
                                 batch_size,
                                 out_channels,
                                 out_height,
                                 out_width);
  } else {
    forward_batch_norm_training(activations,
                                BN_parameters,
                                BN_stats,
                                out_channels,
                                out_height,
                                out_width,
                                batch_size);
  }
  forward_relu(activations, num_activations);
}
