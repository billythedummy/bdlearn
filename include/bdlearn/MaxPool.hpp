#ifndef _BDLEARN_MAXPOOL_H_
#define _BDLEARN_MAXPOOL_H_

#include <cstddef>
#include <iostream>
#include "Halide.h"
#include "bdlearn/macros.hpp"
#include "bdlearn/Layer.hpp"
#include "bdlearn/BMat.hpp"

namespace bdlearn {
    class MaxPool: public Layer {
        public:
            // Constructors
            MaxPool(const int k = 2, const int s = 2): k_(k), s_(s) {};

            // public functions
            void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) override;
            void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) override;
            bufdims calc_out_dim(bufdims in_dims) override;
            void update(float lr) override;

        private:
            int k_; // kernel size
            int s_; // stride
            
            // training vars
            Halide::Buffer<int> max_x_;
            Halide::Buffer<int> max_y_;
    };
}

#endif
