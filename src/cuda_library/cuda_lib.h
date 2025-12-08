#ifndef UNET_CUDA_LIB
#define UNET_CUDA_LIB

enum Initiation_type {
  XAVIER,
  ZERO,
  HE,
  BN, // for batch norm identity initiation
};

float cuda_add_nums();
float *getCudaPointer(int num_floats,
                      Initiation_type i_type = ZERO,
                      int N = 0,
                      int num_weights = 0);

void convolve(const float *weights,
              const float *inputs,
              float *activations,
              int activation_array_length,
              int batch_size,
              int kernel_size,
              int channels_in,
              int channels_out,
              int H_out,
              int W_out,
              int H_in,
              int W_in,
              int padding,
              int stride);
void cuda_concat(float *activations_1,
                 float *activations_2,
                 float *activations,
                 int batch_size,
                 int HxW,
                 int C1,
                 int C2);

void cuda_max_pool(const float *inputs,
                   float *activations,
                   int batch_size,
                   int stride,
                   int kernel_size,
                   int in_channels,
                   int in_height,
                   int in_width);

void cuda_upsample(const float *inputs,
                   const float *weights,
                   float *activations,
                   int scale,
                   int num_in_channels,
                   int num_out_channels,
                   int H_in,
                   int W_in,
                   int batch_size);

void cuda_relu(float *activations, int num_activations);

void cuda_BN_train(float *activations,
                   float *weights,
                   float *BN_batch_stats,
                   float *BN_stats,
                   int num_channels,
                   int H,
                   int W,
                   int batch_size);

void cuda_BN_inference(float *activations,
                       float *weights,
                       float *BN_stats,
                       int H,
                       int W,
                       int num_channels);
void cuda_attention(float *activations_int_1,
                    float *activations_int_2,
                    float *activations,
                    float *weights,
                    float *input_x,
                    float *input_g,
                    int batch_size,
                    int H_x,
                    int W_x,
                    int channels_in_x,
                    int channels_in_g,
                    int int_channels);

#endif
