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

namespace bdlearn {
    class BConvLayer: public Layer {
        public:
        // Constructors
            // default - random initialized
            // k - kernel size, s - stride, in_c - in channels, out_c - out channels
            BConvLayer(const int k, const int in_c, const int out_c, const int s=1, bool train=false)
                : w_(out_c, k*k*in_c), k_(k), s_(s), in_c_(in_c), out_c_(out_c), size_(out_c * k * k * in_c) {
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

        template<typename T>
        T* im2col(
            Halide::Buffer<T>& src,
            int w_src, int h_src, int c_src,
            int p_x, int p_y,
            int s_x, int s_y,
            int k_x, int k_y
        ) {
            
            const int out_height = (h_src + 2*p_y - k_y) / s_y + 1;
            const int out_width = (w_src + 2*p_x - k_x) / s_x + 1;
            const int patch_area = k_x * k_y;
            const int h_im2col = patch_area * c_src;
            const int w_im2col = out_height * out_width;

            T* dest = (T*) calloc(h_im2col * w_im2col, sizeof(T));

            printf("%d %d %d", out_height, out_width, patch_area);
            Halide::Func bim2col("bim2col");
            Halide::Var x, y;
            Halide::Expr c = y / patch_area;
            Halide::Expr pix_index_in_patch = x % patch_area;
            Halide::Expr which_row = x / out_width;
            Halide::Expr which_patch_in_row = x % out_width;
            Halide::Expr top_left_y_index = which_row * s_y - p_y;
            Halide::Expr top_left_x_index = which_patch_in_row * s_x - p_x;
            Halide::Expr y_index = top_left_y_index + (pix_index_in_patch / k_x);
            Halide::Expr x_index = top_left_x_index + (pix_index_in_patch % k_x);
            y_index = Halide::max(y_index, 0);
            y_index = Halide::min(y_index, h_src - 1);
            x_index = Halide::max(x_index, 0);
            x_index = Halide::min(x_index, w_src - 1);

            bim2col(x, y) = print(src(x_index, y_index), x_index, y_index);
            Halide::Func out;
            out(x, y) = bim2col(x, y);
            // Scheudle
            // TO-DO OPTIMIZE ALL THE FANCY STUFF
            out.realize(*dest);
            return dest;
        }
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
            // training vars
            Halide::Buffer<float> prev_in_; // previous input
            std::unique_ptr<float[]> prev_i2c_; // im2col of previous input
            std::unique_ptr<float[]> dw_; // dL/dw
    };
}

#endif