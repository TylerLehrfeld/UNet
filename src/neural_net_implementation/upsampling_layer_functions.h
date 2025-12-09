#ifndef UPSAMPLING_UNET_FUNCS
#define UPSAMPLING_UNET_FUNCS

#include "BN_functions.h"
#include "activation_functions.h"
#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_upsample(float *input,
                                    int batch_size,
                                    bool use_relu_activation,
                                    bool use_batch_norm,
                                    bool inference) {
  const int scale = kernel_size;
  int out_image_size = scale * scale * in_width * in_height;
  int num_activations = out_image_size * in_channels * out_channels;
  cuda_upsample(input,
                parameters,
                activations_int_1,
                scale,
                in_channels,
                out_channels,
                in_height,
                in_width,
                batch_size);

  if(inference) {
    forward_batch_norm_inference(activations_int_1,
                                 activations_int_2,
                                 parameters,
                                 BN_stats,
                                 out_channels,
                                 out_height,
                                 out_width);
  } else {
    forward_batch_norm_training(activations_int_1,
                                activations_int_2,
                                parameters,
                                BN_stats,
                                BN_batch_stats,
                                out_channels,
                                out_height,
                                out_width,
                                batch_size);
  }
  forward_relu(activations_int_2, activations, num_activations);
}

inline void Layer::backward_upsample(float *grad_activations,
                                     float *input,
                                     int batch_size) {
  const int scale = kernel_size;
  int out_image_size = scale * scale * in_width * in_height;
  int num_activations = out_image_size * in_channels * out_channels;

  // Backward through ReLU
  cuda_relu_backward(grad_activations,
                     activations_int_2,
                     grad_activations_int_2,
                     num_activations);

  // Backward through batch norm
  cuda_BN_backward(grad_activations_int_2,
                   activations_int_1,
                   BN_batch_stats,
                   BN_parameters,
                   grad_activations_int_1,
                   dLdW + num_weights_and_biases,
                   out_channels,
                   out_height,
                   out_width,
                   batch_size);

  // Backward through upsample
  cuda_upsample_backward(grad_activations_int_1,
                         input,
                         parameters,
                         dLdX,
                         dLdW,
                         scale,
                         in_channels,
                         out_channels,
                         in_height,
                         in_width,
                         batch_size);
}

#endif
