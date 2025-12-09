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

#endif