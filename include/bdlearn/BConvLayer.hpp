#ifndef _BDLEARN_BCONVLAYER_H_
#define _BDLEARN_BCONVLAYER_H_

#include <cstddef>
#include <iostream>
#include <random>
#include <chrono>
#include "Halide.h"
#include "bdlearn/BMat.hpp"
#include "bdlearn/Layer.hpp"
// for training
#include "bdlearn/BatchBlas.hpp"
#include "schedules/libbatchim2col.h"
#include "schedules/libbatchcol2imaccum.h"
#include "schedules/libbatchmatmulabr.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

namespace bdlearn {
    class BConvLayer: public Layer {
        public:
        // Constructors
            // default - random initialized
            // k - kernel size, s - stride, in_c - in channels, out_c - out channels
            BConvLayer(const int k, const int in_c, const int out_c, const int s=1, bool train=false)
                : w_(out_c, k*k*in_c), k_(k), s_(s), in_c_(in_c), out_c_(out_c), size_(out_c * k * k * in_c),
                lambda_(0.01) {
                float* w_real = new float[size_];
                float n = k * k * out_c;
                // Random init train_w_
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                std::default_random_engine generator (seed);
                std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / n));
                for (int i = 0; i < size_; ++i) w_real[i] = MAX(-0.99, MIN(0.99, dist(generator)));
                // init w_ by sign(train_w_)
                w_.sign(w_real);
                train_w_.reset(w_real);
                if (train) {
                    float* dw = new float[size_];
                    dw_.reset(dw);
                }
            } // VALID PADDING ONLY

        // Destructor
            virtual ~BConvLayer();

        // public functions
            void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) override;
            bufdims calc_out_dim(bufdims in_dims) override;
            void update(float lr) override;
            void load_weights(float* real_weights);
            uint8_t get_w(int x, int y, int in_c, int out_c);
            float get_train_w(int x, int y, int in_c, int out_c);
            float* get_dw();
        
        // friend operators
        friend std::ostream& operator<<(std::ostream& os, const BConvLayer& l);

        int get_cols() {
            return w_.cols();
        }
        int get_rows() {
            return w_.rows();
        }
        private:
            // Halide dims for train_w_: k, k, in_c, out_c
            std::unique_ptr<float[]> train_w_;
            // w_ is already in im2col format i.e. rows = out_c, cols = k*k*in_c
            BMat w_;
            int k_;
            int s_;
            int in_c_;
            int out_c_;
            int size_;
            float lambda_;
            // training vars
            Halide::Buffer<float> prev_in_; // previous input
            std::unique_ptr<float[]> prev_i2c_; // im2col of previous input
            std::unique_ptr<float[]> dw_; // dL/dw
    };

    void BConvIm2Col(Halide::Buffer<float> out, Halide::Buffer<float> in,
                        const int p, const int s, const int k, const int out_width, const int out_height);
}

#endif