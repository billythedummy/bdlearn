#ifndef _BDLEARN_LAYER_H_
#define _BDLEARN_LAYER_H_

#include <cstddef>
#include <iostream>
#include <memory>
#include "Halide.h"

namespace bdlearn {
    class Layer {
        public:
        // Destructor
            virtual ~Layer();

        // public functions
            virtual void forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in); // training
            virtual void forward_i(Halide::Buffer<float>* out, Halide::Buffer<float> in); // inference
            virtual void backward(Halide::Buffer<float>* out, Halide::Buffer<float> ppg); // previous partial gradients
        
        // friend operators
        friend std::ostream& operator<<(std::ostream& os, const Layer& l);


        private:
            Layer& operator=(const Layer& ref) = delete;
    };
}

#endif