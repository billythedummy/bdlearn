#ifndef _BDLEARN_BCONVLAYER_H_
#define _BDLEARN_BCONVLAYER_H_

#include <cstddef>
#include <iostream>
#include <random>
#include <chrono>
#include "Halide.h"
#include "bdlearn/BMat.hpp"
#include "bdlearn/Layer.hpp"

namespace bdlearn {
    class BConvLayer: public Layer {
        public:
        // Constructors
            // default - random initialized
            // k - kernel size, s - stride, in_c - in channels, out_c - out channels
            BConvLayer(int k, int s, int in_c, int out_c) : w_(out_c, k*k*in_c) {
                k_ = k;
                s_ = s;
                in_c_ = in_c;
                out_c_ = out_c;
                size_ = out_c * k * k * in_c;
                float* w_real = new float[size_];
                float n = k * k * out_c;
                // Random init train_w_
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                std::default_random_engine generator (seed);
                std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / n));
                for (int i = 0; i < size_; ++i) w_real[i] = dist(generator);
                // init w_ by sign(train_w_)
                w_.sign(w_real);
                train_w_.reset(w_real);
            } // VALID PADDING ONLY

        // Destructor
            virtual ~BConvLayer();

        // public functions
            void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) override;
            //void load_weights();
            uint8_t get_w(int x, int y, int in_c, int out_c);
            float get_train_w(int x, int y, int in_c, int out_c);
        
        // friend operators
        friend std::ostream& operator<<(std::ostream& os, const BConvLayer& l);

        private:
            std::unique_ptr<float[]> train_w_;
            // w_ is already in im2col format i.e. rows = out_c, cols = k*k*in_c
            BMat w_;
            int k_;
            int s_;
            int in_c_;
            int out_c_;
            int size_;
    };
}

#endif