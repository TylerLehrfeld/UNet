#ifndef UNET_CUDA_LIB
#define UNET_CUDA_LIB

enum Initiation_type {
  XAVIER,
  ZERO,
  HE,
};

float cuda_add_nums();
float *getCudaPointer(int num_floats, Initiation_type i_type = ZERO, int N = 0,
                      int num_weights = 0);
void convolve(const float *weights, const float *inputs, float *activations,
              int activation_array_length, int batch_size, int kernel_size,
              int channels_in, int channels_out, int H_out, int W_out, int H_in,
              int W_in, int padding, int stride);
#endif
