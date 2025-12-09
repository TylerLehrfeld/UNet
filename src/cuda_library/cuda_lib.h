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

void cuda_relu(float *inputs, float *activations, int num_activations);

void cuda_sigmoid(float *inputs, float *activations, int num_activations);

void cuda_BN_train(float *inputs,
                   float *activations,
                   float *weights,
                   float *BN_batch_stats,
                   float *BN_stats,
                   int num_channels,
                   int H,
                   int W,
                   int batch_size);

void cuda_BN_inference(float *inputs,
                       float *activations,
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

float cuda_dice_score(float *mask,
                      float *prediction,
                      int H,
                      int W,
                      int batch_size,
                      float *loss_values);

void cuda_dice_loss_backward(float *mask,
                             float *prediction,
                             float *loss_values,
                             float *dLdY,
                             int H,
                             int W,
                             int batch_size);

void cuda_relu_backward(float *grad_activations,
                        float *inputs,
                        float *grad_inputs,
                        int num_activations);

void cuda_sigmoid_backward(float *grad_activations,
                           float *activations,
                           float *grad_inputs,
                           int num_activations);

void convolve_backward(float *grad_activations,
                       float *inputs,
                       float *weights,
                       float *grad_inputs,
                       float *grad_weights,
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
void cuda_BN_backward(float *grad_activations,
                      float *inputs,
                      float *BN_batch_stats,
                      float *weights,
                      float *grad_inputs,
                      float *grad_weights,
                      int num_channels,
                      int H,
                      int W,
                      int batch_size);

void cuda_concat_backward(
    float *dLdY, float *dLdX, int H, int W, int C1, int C2, int batch_size);

void cudaSetZero(float *arr, int num_floats);

void cuda_max_pool_backward(const float *grad_activations,
                            const float *inputs,
                            const float *activations,
                            float *grad_inputs,
                            int batch_size,
                            int stride,
                            int kernel_size,
                            int in_channels,
                            int in_height,
                            int in_width);

#endif
