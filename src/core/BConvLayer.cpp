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
        prev_in_ = in;
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
        // some constants for use
        const int batches = ppg.dim(3).extent();
        const int out_width = ppg.dim(0).extent();
        const int out_height = ppg.dim(1).extent();
        const int total_space = out_width * out_height;
        float* ppg_begin = ppg.get()->begin();
        Halide::Buffer<float> ppg_re(ppg_begin, total_space,
                                    ppg.dim(2).extent(), batches);
        // ppg_re's dimensions is (total_space, out_c, batch) in Halide dims
        const int kkic = k_ * k_ * in_c_;
        
        // dl/dsign(w) algo
        Halide::Var x, y, c, n;
        float dsignw [kkic*out_c_*batches];
        Halide::Buffer<float> dsignwbatch_view(dsignw, kkic, out_c_, batches);
        Halide::Buffer<float> prev_i2c_view(prev_i2c_.get(), total_space, kkic, batches);
        BatchMatMul_BT(dsignwbatch_view, ppg_re, prev_i2c_view);
        Halide::Buffer<float> dw_view(dw_.get(), kkic, out_c_);
        Halide::Func dsignw_f;
        Halide::RDom b(0, batches);
        dsignw_f(x, y) = 0.0f;
        dsignw_f(x, y) += dsignwbatch_view(x, y, b);
        // dl/dsign(w) schedule
        dsignw_f.realize(dw_view);

        // dl/dw algo
        Halide::Buffer<float> w_view(train_w_.get(), kkic, out_c_);
        Halide::Func dw_ste_f; // dsign(w)/dw = w{|w| < 1}
        dw_ste_f(x, y) = dw_view(x, y) * Halide::select(Halide::abs(w_view(x, y)) < 1, w_view(x, y), 0);
        // dl/dw schedule
        dw_ste_f.realize(dw_view);

        // dl/dsign(x) algo
        float dcol [batches * kkic * total_space];
        Halide::Buffer<float> dcol_view(dcol, total_space, kkic, batches, "dcol_view");
        BatchMatMul_ATBr(dcol_view, w_view, ppg_re);
        BatchCol2ImAccum(out, dcol_view, 0, s_, k_, out_width, out_height);

        // dl/dx algo
        Halide::Func dx_ste_f;
        Halide::Expr x_ste_sel = Halide::select(Halide::abs(prev_in_(x, y, c, n)) < 1, prev_in_(x, y, c, n), 0);
        dx_ste_f(x, y, c, n) = out(x, y, c, n) * x_ste_sel;
        // dl/dx schedule
        dx_ste_f.realize(out);
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

    float* BConvLayer::get_dw() {
        return dw_.get();
    }

    float BConvLayer::get_train_w(int x, int y, int in_c, int out_c) {
        return train_w_.get()[out_c * k_ * k_ * in_c_
                                + in_c * k_ * k_
                                + y * k_
                                + x];
    }
}