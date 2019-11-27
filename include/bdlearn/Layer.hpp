#ifndef _BDLEARN_LAYER_H_
#define _BDLEARN_LAYER_H_

#include <cstddef>
#include <iostream>
#include <memory>
#include "Halide.h"
#include "bdlearn/BMat.hpp"

namespace bdlearn {
    class Layer {
        public:
        // public functions
            virtual void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) = 0; // training
            virtual void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) = 0; // inference
            virtual void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) = 0; // previous partial gradients
        
        // friend operators
        //friend std::ostream& operator<<(std::ostream& os, const Layer& l);


        private:
            Layer& operator=(const Layer& ref) = delete;
    };
}

#endif