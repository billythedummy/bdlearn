#ifndef _BDLEARN_POOL_H_
#define _BDLEARN_POOL_H_

#include <cstddef>
#include <iostream>
#include "Halide.h"
#include "bdlearn/macros.hpp"
#include "bdlearn/Layer.hpp"

namespace bdlearn {
    class Pool: public Layer {
        public:
            // Constructors
            Pool();

            // Destructors
            virtual ~Pool();

            // public functions
            void forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float>* out, Halide::Buffer<float> ppg) override;

        private:
            int channels_;
            std::unique_ptr<float[]> gamma_; // scale
            std::unique_ptr<float[]> beta_; // translation
            std::unique_ptr<float[]> r_mean_; // running mean
            std::unique_ptr<float[]> r_var_; // running variance
            std::unique_ptr<float[]> mu_; // mean from prev input
            std::unique_ptr<float[]> var_; // variance from prev input
    }
}