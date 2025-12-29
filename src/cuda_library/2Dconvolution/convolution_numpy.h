
#include "cuda_lib.h"
#include <cassert>
#include <iostream>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

class Convolution {
private:
  template <typename T>
  T *getPointerFromNumpy(py::array_t<T> np_arr, py::buffer_info &info) {
    info = np_arr.request();
    return static_cast<T *>(info.ptr);
  }
  template <typename T> T *getCudaPointerCopy(T *orig, int num_elements) {
    T *result = cudaGetPointer<T>(num_elements);
    cudaLibCopyToDevice<T>(orig, result, num_elements);
    return result;
  }

public:
  py::array_t<float> train(py::array_t<float> input_image,
                           py::array_t<float> y_image,
                           py::array_t<float> weights,
                           py::array_t<float> biases,
                           int padding,
                           int stride,
                           float learning_rate) {
    std::cout << "starting train" << std::endl;
    py::buffer_info input_info;
    py::buffer_info output_info;
    py::buffer_info weight_info;
    py::buffer_info bias_info;
    float *input = getPointerFromNumpy(input_image, input_info);
    float *y = getPointerFromNumpy(y_image, output_info);
    float *kernels = getPointerFromNumpy(weights, weight_info);
    float *bias_ptr = getPointerFromNumpy(biases, bias_info);
    assert(input_info.shape.size() == 3);
    int C_in = input_info.shape[0];
    int H_in = input_info.shape[1];
    int W_in = input_info.shape[2];
    int input_size = C_in * H_in * W_in;
    assert(output_info.shape.size() == 3);
    int C_out = output_info.shape[0];
    int H_out = output_info.shape[1];
    int W_out = output_info.shape[2];
    int output_size = C_out * H_out * W_out;
    assert(C_in == weight_info.shape[0]);
    assert(C_out == weight_info.shape[1]);
    int kh = weight_info.shape[2];
    int kw = weight_info.shape[3];
    int num_weights = kh * kw * C_in * C_out;
    int num_weights_and_biases = num_weights + C_out;
    assert(H_out == (H_in + 2 * padding - kh) / stride + 1);
    assert(W_out == (W_in + 2 * padding - kw) / stride + 1);
    assert(bias_info.shape[0] == C_out);
    if(kh != kw || kh % 2 == 0) {
      throw std::runtime_error("kernel must be square and odd.");
    }

    std::cout << "asserts passed" << std::endl;
    float *dInput = getCudaPointerCopy<float>(input, input_size);
    float *dY = getCudaPointerCopy(y, output_size);
    float *dWeights = cudaGetPointer<float>(num_weights_and_biases);
    float *dOutput_image = cudaGetPointer<float>(output_size);
    float *dLdY = cudaGetPointer<float>(output_size);
    float *dLdX = cudaGetPointer<float>(input_size);
    float *dLdW = cudaGetPointer<float>(num_weights_and_biases);
    float *adam_parameters = cudaGetPointer<float>(num_weights_and_biases * 2);
    cudaLibCopyToDevice(kernels, dWeights, num_weights);
    cudaLibCopyToDevice(bias_ptr, dWeights + num_weights, C_out);
    py::array_t<float> out_numpy_weights =
        py::array_t<float>(num_weights_and_biases);

    py::buffer_info out_info = out_numpy_weights.request();

    float *output_weights = static_cast<float *>(out_info.ptr);

    std::cout << "pointers allocated" << std::endl;
    for(int t = 0; t < 100; t++) {
      if(cudaHasNans(dWeights, num_weights_and_biases))
        std::cout << "has nans weights before convolve: " << t << std::endl;

      cudaConvolve(dWeights,
                   dInput,
                   dOutput_image,
                   output_size,
                   1,
                   kh,
                   C_in,
                   C_out,
                   H_out,
                   W_out,
                   H_in,
                   W_in,
                   padding,
                   stride);
      if(cudaHasNans(dOutput_image, output_size))
        std::cout << "has nans output: " << t << std::endl;
      cudaDifferenceLossBackward(dY, dOutput_image, dLdY, output_size);
      if(cudaHasNans(dLdY, output_size))
        std::cout << "has nans difference loss: " << t << std::endl;
      cudaConvolveBackward(dLdY,
                           dInput,
                           dWeights,
                           dLdX,
                           dLdW,
                           1,
                           kh,
                           C_in,
                           C_out,
                           H_out,
                           W_out,
                           H_in,
                           W_in,
                           padding,
                           stride);
      if(cudaHasNans(dLdW, num_weights_and_biases))
        std::cout << "has nans dLdW: " << t << std::endl;

      // cudaUpdateWeights(dWeights, dLdW, 0.39f, num_weights_and_biases);
      cudaUpdateWeights(dWeights,
                        dLdW,
                        adam_parameters,
                        learning_rate,
                        num_weights_and_biases,
                        t + 1);
      if(cudaHasNans(dWeights, num_weights_and_biases))
        std::cout << "has nans weights: " << t << std::endl;

      cudaLibCopyToHost(output_weights, dWeights, num_weights_and_biases);

      // Copy dLdW and dLdY to host as well
      std::vector<float> host_dLdW(num_weights_and_biases);
      cudaLibCopyToHost(host_dLdW.data(), dLdW, num_weights_and_biases);

      std::vector<float> host_dLdY(output_size);
      cudaLibCopyToHost(host_dLdY.data(), dLdY, output_size);

      std::vector<float> host_dOutput(output_size);
      cudaLibCopyToHost(host_dOutput.data(), dOutput_image, output_size);

      std::vector<float> host_input(input_size);
      cudaLibCopyToHost(host_input.data(), dInput, input_size);
      if(t % 1 == -1) {
        std::cout << "=== Iteration " << t << " ===" << std::endl;

        std::cout << "[DEBUG] Weights + bias:" << std::endl;
        for(int i = 0; i < kh; ++i) {
          for(int j = 0; j < kw; ++j)
            std::cout << output_weights[i * kw + j] << " ";
          std::cout << std::endl;
        }
        std::cout << "Bias: " << output_weights[num_weights] << std::endl;

        std::cout << "[DEBUG] Weight gradients dLdW: ";
        for(int i = 0; i < num_weights_and_biases; i++)
          std::cout << host_dLdW[i] << " ";
        std::cout << std::endl;

        std::cout << "[DEBUG] Output gradients dLdY: ";
        for(int i = 0; i < output_size; i++)
          std::cout << host_dLdY[i] << " ";
        std::cout << std::endl;

        std::cout << "[DEBUG] Output image: ";
        for(int i = 0; i < output_size; i++)
          std::cout << host_dOutput[i] << " ";
        std::cout << std::endl;

        std::cout << "[DEBUG] Input image: ";
        for(int i = 0; i < input_size; i++)
          std::cout << host_input[i] << " ";
        std::cout << std::endl;
      }
    }
    cudaLibFree(dInput);
    cudaLibFree(dY);
    cudaLibFree(dWeights);
    cudaLibFree(dOutput_image);
    cudaLibFree(dLdY);
    cudaLibFree(dLdX);
    cudaLibFree(dLdW);
    cudaLibFree(adam_parameters);

    return out_numpy_weights;
  }
  py::array_t<float> convolution2D(py::array_t<float> image,
                                   py::array_t<float> kernel,
                                   py::array_t<float> biases,
                                   int padding,
                                   int stride) {
    py::buffer_info img_info = image.request();
    py::buffer_info kernel_info = kernel.request();
    py::buffer_info bias_info = biases.request();
    if(img_info.shape.size() != 3) {
      throw std::runtime_error("Only accepting single images");
    }
    if(kernel_info.shape.size() != 4) {
      std::stringstream ss;
      ss << "Only accepting kernels with the format [C_in][C_out][H][W]. Found "
            "size of "
         << kernel_info.shape.size() << std::endl;
      throw std::runtime_error(ss.str());
    }
    float *img_ptr = static_cast<float *>(img_info.ptr);
    float *kernel_ptr = static_cast<float *>(kernel_info.ptr);

    float *bias_ptr = static_cast<float *>(bias_info.ptr);
    int C_in = img_info.shape[0];
    int H = img_info.shape[1];
    int W = img_info.shape[2];
    int image_size = C_in * H * W;
    if(C_in != kernel_info.shape[0]) {
      std::stringstream ss;
      ss << "Kernel C_in must match input C_in: " << C_in << ", "
         << kernel_info.shape[0] << std::endl;
      throw std::runtime_error(ss.str());
    }
    int C_out = kernel_info.shape[1];
    int kh = kernel_info.shape[2];
    int kw = kernel_info.shape[3];
    assert(C_out = bias_info.shape[0]);

    if(kh != kw || kh % 2 == 0) {
      throw std::runtime_error("kernel must be square and odd.");
    }
    // get a output_image pointer of the same size
    py::array_t<float> out_numpy_image = py::array_t<float>(image_size);

    py::buffer_info out_info = out_numpy_image.request();

    float *output_image = static_cast<float *>(out_info.ptr);
    int H_out = (H + 2 * padding - kh) / stride + 1;
    int W_out = (W + 2 * padding - kw) / stride + 1;
    float *dImage = cudaGetPointer<float>(image_size);
    float *dKernel = cudaGetPointer<float>(kh * kw * C_in * C_out + C_out);
    cudaLibCopyToDevice(img_ptr, dImage, image_size);
    cudaLibCopyToDevice(kernel_ptr, dKernel, kh * kw * C_out * C_in);
    cudaLibCopyToDevice(bias_ptr, dKernel + kh * kw * C_in * C_out, C_out);
    float *dOutput_image = cudaGetPointer<float>(image_size);
    std::cout << "in convolution" << std::endl;
    cudaConvolve(dKernel,
                 dImage,
                 dOutput_image,
                 image_size,
                 1,
                 kh,
                 C_in,
                 C_out,
                 H_out,
                 W_out,
                 H,
                 W,
                 padding,
                 stride);

    std::cout << "out of convolution" << std::endl;
    cudaLibCopyToHost(output_image, dOutput_image, image_size);
    cudaLibFree(dImage);
    cudaLibFree(dKernel);
    cudaLibFree(dOutput_image);
    return out_numpy_image;
  }
};
