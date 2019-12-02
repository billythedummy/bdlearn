#ifndef _BDLEARN_LOSS_LAYER_H_
#define _BDLEARN_LOSS_LAYER_H_

#include "Halide.h"

namespace bdlearn {
    class LossLayer {
        public:
        // public functions
            virtual float forward_t(Halide::Buffer<float> in, Halide::Buffer<float> target, void* args=nullptr) = 0; // returns loss
            virtual void backward(Halide::Buffer<float> out_ppg) = 0;
        // friend operators
        //friend std::ostream& operator<<(std::ostream& os, const Layer& l);

        private:
            LossLayer& operator=(const LossLayer& ref) = delete;
    };
}

#endif