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
            BConvLayer(size_t k, size_t s, size_t in_c, size_t out_c) : w_(out_c, k*k*in_c) {
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
                for (size_t i = 0; i < size_; ++i) w_real[i] = dist(generator);
                // init w_ by sign(train_w_)
                w_.sign(w_real);
                train_w_.reset(w_real);
            } // VALID PADDING ONLY

        // Destructor
            virtual ~BConvLayer();

        // public functions
            void forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float>* out, Halide::Buffer<float> ppg) override;
            //void load_weights();
            uint8_t get_w(size_t x, size_t y, size_t in_c, size_t out_c);
            float get_train_w(size_t x, size_t y, size_t in_c, size_t out_c);
        
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
            T* dest = new T[w_src*h_src*c_src];
            
            const int out_height = (h_src + 2*p_y - k_y) / s_y + 1;
            const int out_width = (w_src + 2*p_x - k_x) / s_x + 1;
            const int patch_area = k_x * k_y;
            const int h_im2col = patch_area * c_src;
            const int w_im2col = out_height * out_width;

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
            bim2col(x, y) = src(x_index, y_index);
            
            Halide::Func out;
            out(x, y) = bim2col(x, y);
            // Scheudle
            // TO-DO OPTIMIZE ALL THE FANCY STUFF
            out.realize(*dest);
            return dest;
        }
        private:
            std::unique_ptr<float[]> train_w_;
            // w_ is already in im2col format i.e. rows = out_c, cols = k*k*in_c
            BMat w_;
            size_t k_;
            size_t s_;
            size_t in_c_;
            size_t out_c_;
            size_t size_;
    };
}

#endif