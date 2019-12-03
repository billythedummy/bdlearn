#ifndef _BDLEARN_LAYER_H_
#define _BDLEARN_LAYER_H_

#include "Halide.h"
#include "bdlearn/BufDims.hpp"
#include "bdlearn/utils.hpp"

namespace bdlearn {
    class Layer {
        public:
        // public functions
            virtual void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) = 0; // training
            virtual void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) = 0; // inference
            virtual void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) = 0; // previous partial gradients
            virtual bufdims calc_out_dim(bufdims in_dims) = 0;
            virtual void update(float lr) = 0;
            virtual void save_layer(std::ofstream& fout) = 0;
            virtual void load_layer(std::ifstream& fin) = 0;
        // friend operators
        //friend std::ostream& operator<<(std::ostream& os, const Layer& l);

        private:
            Layer& operator=(const Layer& ref) = delete;
    };
}

#endif