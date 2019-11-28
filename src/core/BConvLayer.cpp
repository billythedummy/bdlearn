#include "bdlearn/BConvLayer.hpp"

namespace bdlearn {
    // Constructors

    // Destructor

    BConvLayer::~BConvLayer() {
        train_w_.reset();
        prev_i2c_.reset();
    }

    // public functions

    void BConvLayer::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // calculate dims
        const int cols = in.dim(0).extent();
        const int rows = in.dim(1).extent();
        const int batches = in.dim(3).extent();
        const int out_height = (rows - k_) / s_ + 1;
        const int out_width = (cols - k_) / s_ + 1;
        const int patch_area = k_* k_;
        const int h_im2col = patch_area * in_c_;
        const int w_im2col = out_height * out_width;
        // Sign input
        Halide::Var x, y, c, n;
        Halide::Func sign_in_f;
        sign_in_f(x, y, c, n) = Halide::select(in(x, y, c, n) >= 0, 1.0f, -1.0f);
        float sign_in [cols*rows*in_c_*batches];
        Halide::Buffer<float> sign_in_view(sign_in, cols, rows, in_c_, batches);
        sign_in_f.realize(sign_in_view);
        // Sign W
        Halide::Func sign_w_f;
        Halide::Buffer<float> train_w_view(train_w_.get(), k_*k_*in_c_, out_c_);
        sign_w_f(x, y) = Halide::select(train_w_view(x, y) >= 0, 1.0f, -1.0f);
        float sign_w [k_*k_*in_c_*out_c_];
        Halide::Buffer<float> sign_w_view(sign_w, k_*k_*in_c_, out_c_);
        sign_w_f.realize(sign_w_view);
        // im2col input
        float* sign_in_im2col = new float[h_im2col * w_im2col * batches];
        Halide::Buffer<float> sign_in_im2col_view (sign_in_im2col, w_im2col, h_im2col, batches);
        BatchIm2Col(sign_in_im2col_view, sign_in_view, 0, s_, k_, out_width, out_height);
        prev_i2c_.reset(sign_in_im2col);
        // Matmul
        float* out_begin = out.get()->begin(); // this is super hacky i know
        Halide::Buffer<float> out_view(out_begin, w_im2col, out_c_, batches);
        BatchMatMul_ABr(out_view, sign_w_view, sign_in_im2col_view);
    }

    void BConvLayer::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // TO-DO
        return;
    }

    void BConvLayer::backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) {
        // dl/dsign(w)
        const int batches = ppg.dim(3).extent();
        const int total_space = ppg.dim(0).extent()*ppg.dim(1).extent();
        float* ppg_begin = ppg.get()->begin(); // this is super hacky i know
        Halide::Buffer<float> ppg_re(ppg_begin, total_space,
                                    ppg.dim(2).extent(), ppg.dim(3).extent());
        // ppg_re's dimensions is now [batch, out_c, out_height*out_width] or (out_width*out_height, out_c, batch) in Halide dims
        float dsignw [k_*k_*in_c_*out_c_*batches];
        Halide::Buffer<float> dsignw_view(dsignw, k_*k_*in_c_, out_c_, batches);
        Halide::Buffer<float> prev_i2c_view(prev_i2c_.get(), total_space, k_*k_*in_c_, batches);
        BatchMatMul_BT(dsignw_view, ppg_re, prev_i2c_view);
    }

    void BConvLayer::load_weights(float* real_weights) {
        if (!train_w_) {
            train_w_.reset(new float[size_]);
        }
        for (int i = 0; i < size_; ++i) train_w_.get()[i] = real_weights[i];
        w_.sign(train_w_.get());
    }

    uint8_t BConvLayer::get_w(int x, int y, int in_c, int out_c) {
        return w_.get(out_c, in_c * k_ * k_ + y * k_ + x);
    }

    float BConvLayer::get_train_w(int x, int y, int in_c, int out_c) {
        return train_w_.get()[out_c * k_ * k_ * in_c_
                                + in_c * k_ * k_
                                + y * k_
                                + x];
    }
}