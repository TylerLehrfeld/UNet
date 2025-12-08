#ifndef BN_UNET_FUNCS
#define BN_UNET_FUNCS
#include "cuda_lib.h"

inline void forward_batch_norm_inference(float *activations,
                                         float *weights,
                                         float *BN_stats,
                                         int num_channels,
                                         int H,
                                         int W) {
  cuda_BN_inference(activations, weights, BN_stats, H, W, num_channels);
}

inline void forward_batch_norm_training(float *activations,
                                        float *weights,
                                        float *BN_stats,
                                        float *BN_batch_stats,
                                        int num_channels,
                                        int H,
                                        int W,
                                        int batch_size) {
  cuda_BN_train(activations,
                weights,
                BN_batch_stats,
                BN_stats,
                num_channels,
                H,
                W,
                batch_size);
}

#endif // !BN_UNET_FUNCS
