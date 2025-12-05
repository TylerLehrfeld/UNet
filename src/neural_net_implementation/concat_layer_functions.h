#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_concat(float *activations_1,
                                  float *activations_2,
                                  int batch_size) {
  cuda_concat(activations_1,
              activations_2,
              activations,
              batch_size,
              input_shape[2] * input_shape[3],
              input_shape[0],
              input_shape[1]);
}
