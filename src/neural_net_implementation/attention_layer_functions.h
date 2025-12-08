#include "cuda_lib.h"
#include "layer.h"

inline void
Layer::forward_attention(float *input_x, float *input_g, int batch_size) {

  cuda_attention(activations_int_1,
                 activations_int_2,
                 activations,
                 parameters,
                 input_x,
                 input_g,
                 batch_size,
                 in_height,
                 in_width,
                 in_channels,
                 input_shape[3],
                 in_channels / 2);
}
