#ifndef _BDLEARN_SOFTMAX_CROSS_ENTROPY_H_
#define _BDLEARN_SOFTMAX_CROSS_ENTROPY_H_

#include "Halide.h"
#include "bdlearn/LossLayer.hpp"

namespace bdlearn {
    class SoftmaxCrossEntropy : public LossLayer {
        public:
        // Constructors
            // default - gamma, rvar, 1, beta, rmean 0
            SoftmaxCrossEntropy() {};

        // Destructor
            virtual ~SoftmaxCrossEntropy();

        // public functions
            float forward_t(Halide::Buffer<float> in, Halide::Buffer<float> one_hot) override; // returns loss
            void backward(Halide::Buffer<float> out_ppg) override;
            float* get_q(void);

        private:
            std::unique_ptr<float[]> q_; // Halide dims: channels, batch. softmax output: e^x / sum(e^x)
            Halide::Buffer<float> one_hot_; // Halide dims: channels, batch
    };
}

#endif