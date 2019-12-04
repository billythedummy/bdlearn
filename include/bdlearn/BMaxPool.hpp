#ifndef _BDLEARN_BMAXPOOL_H_
#define _BDLEARN_BMAXPOOL_H_

#include <cstddef>
#include <iostream>
#include "Halide.h"
#include "bdlearn/macros.hpp"
#include "bdlearn/Layer.hpp"
#include "bdlearn/BMat.hpp"

namespace bdlearn {
    class BMaxPool: public Layer {
        public:
            // Constructors
            BMaxPool(const int k = 2, const int s = 2): k_(k), s_(s), has_batches(false) {};

            // public functions
            void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in); // training
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in);
            void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg);
        
        private:
            int k_; // kernel size
            int s_; // stride
            bool has_batches;
            
            // training vars
            Halide::Buffer<float> prev_in_; // previous input
            Halide::Buffer<int> max_x_;
            Halide::Buffer<int> max_y_;
            Halide::Buffer<float> max_val_;
    };
}

#endif
