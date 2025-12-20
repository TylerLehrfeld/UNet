#ifndef UNET_CUDA_LIB
#define UNET_CUDA_LIB

enum Initiation_type {
  XAVIER,
  ZERO,
  HE,
  BN, // for batch norm identity initiation
};

void cuda_add_nums(float *a, float *b, int num_floats);
float *getCudaPointer(int num_floats,
                      Initiation_type i_type = ZERO,
                      int N = 0,
                      int num_weights = 0);
template <typename T>
void cudaConvolve(const T *__restrict__ weights,
                  const T *__restrict__ inputs,
                  T *__restrict__ activations,
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
template <typename T>
void cudaMaxPool(const T *__restrict__ inputs,
                 T *__restrict__ activations,
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

float cuda_dice_score(const float *mask,
                      const float *prediction,
                      int H,
                      int W,
                      int batch_size,
                      float *loss_values);

void cuda_dice_loss_backward(const float *mask,
                             const float *prediction,
                             const float *loss_values,
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
template <typename T>
void cudaConvolveBackward(T *grad_activations,
                          T *inputs,
                          T *weights,
                          T *grad_inputs,
                          T *grad_weights,
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

template <typename T>
void cudaMaxPoolBackward(const T *__restrict__ grad_activations,
                         const T *__restrict__ inputs,
                         const T *__restrict__ activations,
                         T *__restrict__ grad_inputs,
                         int batch_size,
                         int stride,
                         int in_channels,
                         int in_height,
                         int in_width);
void cuda_upsample_backward(const float *grad_activations,
                            const float *inputs,
                            const float *weights,
                            float *grad_inputs,
                            float *grad_weights,
                            int scale,
                            int channels_in,
                            int channels_out,
                            int H_in,
                            int W_in,
                            int batch_size);

void cuda_attention_backward(float *grad_activations,
                             float *activations_int_1,
                             float *activations_int_2,
                             float *input_x,
                             float *input_g,
                             float *weights,
                             float *grad_activations_int_1,
                             float *grad_activations_int_2,
                             float *grad_input_x,
                             float *grad_input_g,
                             float *grad_weights,
                             int batch_size,
                             int H_x,
                             int W_x,
                             int channels_in_x,
                             int channels_in_g,
                             int int_channels);
void cuda_update_weights(float *weights,
                         float *dLdW,
                         float *adam_parameters,
                         float learning_rate,
                         int num_weights,
                         int t);

void cudaLibFree(float *dPointer);
void cuda_threshold(float *x, float threshold, int num_floats);
float *cudaToCPU(float *dPointer, int num_floats);
void cuda_gpu_reset();
void cuda_check_err();
float cudaValToCPU(float *dPointer, int ind);
void cuda_lib_copy_device(float *orig, int num_floats, float *copy);

void cuda_lib_copy_to_device(float *cpu, float *gpu, int num_floats);

bool has_nans(const float *d_pointer, int num_floats);
#endif
