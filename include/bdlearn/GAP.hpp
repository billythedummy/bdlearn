#ifndef _BDLEARN_GAP_H_
#define _BDLEARN_GAP_H_

#include <cstddef>
#include <iostream>
#include "Halide.h"
#include "bdlearn/macros.hpp"
#include "bdlearn/Layer.hpp"

namespace bdlearn {
    class GAP: public Layer {
        public:
        // Constructors
            // default - gamma, rvar, 1, beta, rmean 0
            GAP(bool training=false) {};

        // Destructor
            virtual ~GAP(){};

        // public functions
            void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) override;
            bufdims calc_out_dim(bufdims in_dims) override;
            void update(float lr) override;
    };
}

#endif