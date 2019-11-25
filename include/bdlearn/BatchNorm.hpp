#ifndef _BDLEARN_BATCHNORM_H_
#define _BDLEARN_BATCHNORM_H_

#include <cstddef>
#include <iostream>
#include "Halide.h"
#include "bdlearn/macros.hpp"
#include "bdlearn/Layer.hpp"

namespace bdlearn {
    class BatchNorm: public Layer {
        public:
        // Constructors
            // default - gamma 1, beta 0
            BatchNorm(int channels);

        // Destructor
            virtual ~BatchNorm();

        // public functions
            void forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float>* out, Halide::Buffer<float> ppg) override;
            void set_gamma(float* data);
            void set_beta(float* data);

        private:
            int channels_;
            std::unique_ptr<float[]> gamma_; // scale
            std::unique_ptr<float[]> beta_; // translation
            std::unique_ptr<float[]> mu_; // mean from prev input
            std::unique_ptr<float[]> var_; // variance from prev input
    };
}

#endif