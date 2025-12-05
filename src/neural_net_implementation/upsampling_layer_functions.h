#include "cuda_lib.h"
#include "layer.h"
#include "BN_functions.h"
#include "activation_functions.h"
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
                activations,
                scale,
                in_channels,
                out_channels,
                in_height,
                in_width,
                batch_size);

  if(inference) {

  } else {
  }
  forward_relu(activations, num_activations);
}
