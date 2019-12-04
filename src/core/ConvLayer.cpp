#include "bdlearn/ConvLayer.hpp"

namespace bdlearn {
    // Constructors

    // Destructor

    ConvLayer::~ConvLayer() {
        train_w_.reset();
        prev_i2c_.reset();
    }

    // public functions

    void ConvLayer::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
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
        // im2col input
        float* sign_in_im2col = new float[h_im2col * w_im2col * batches];
        Halide::Buffer<float> sign_in_im2col_view (sign_in_im2col, w_im2col, h_im2col, batches);
        libbatchim2col(*in.get(), 0, s_, k_, out_width, out_height, *sign_in_im2col_view.get());
        prev_i2c_.reset(sign_in_im2col);
        // Matmul
        float* out_begin = out.get()->begin(); // this is super hacky i know
        Halide::Buffer<float> out_view(out_begin, w_im2col, out_c_, batches);
        Halide::Buffer<float> w_view(train_w_.get(), k_*k_*in_c_, out_c_);
        libbatchmatmulabr(*w_view.get(), *sign_in_im2col_view.get(), *out_view.get());
    }

    void ConvLayer::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // TO-DO
        /*
        const int cols = in.dim(0).extent();
        const int rows = in.dim(1).extent();

        const int out_height = (rows - k_) / s_ + 1;
        const int out_width = (cols - k_) / s_ + 1;
        const int patch_area = k_* k_;
        const int h_im2col = patch_area * in_c_;
        const int w_im2col = out_height * out_width;

        // Put in into im2col
        float* in_im2col = new float[h_im2col * w_im2col];
        Halide::Buffer<float> in_im2col_view (in_im2col, w_im2col, h_im2col);
        ConvIm2Col(in_im2col_view, in, 0, s_, k_, out_width, out_height);
        // Matmul weights with BMat
        float* out_begin = out.get()->begin(); // this is super hacky i know
        Halide::Buffer<float> out_view(out_begin, in_mat.cols(), w_.rows());

        prev_in_ = in;
        prev_i2c_.reset(in_im2col);*/
    }

    void ConvLayer::backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) {
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
        float* dsignw = new float [kkic*out_c_*batches];
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

        Halide::Buffer<float> w_view(train_w_.get(), kkic, out_c_);

        // dl/dsign(x) algo
        float* dcol = new float [batches * kkic * total_space];
        Halide::Buffer<float> dcol_view(dcol, total_space, kkic, batches, "dcol_view");
        BatchMatMul_ATBr(dcol_view, w_view, ppg_re);
        libbatchcol2imaccum(*dcol_view.get(), 0, s_, k_, out_width, out_height, *out.get());

        // free
        delete[] dsignw;
        delete[] dcol;
    }

    bufdims ConvLayer::calc_out_dim(bufdims in_dims) {
        assert(in_dims.c == in_c_);
        return bufdims {
            (in_dims.w - k_) / s_ + 1,
            (in_dims.h - k_) / s_ + 1,
            out_c_
        };
    }

    void ConvLayer::update(float lr) {
        /*
        std::cout << "W before: " << std::endl;
        std::cout << std::endl;
        for (int i = 0; i < out_c_; ++i) {
            for (int j = 0; j < k_*k_*in_c_; ++j) {
                std::cout << train_w_[i*k_*k_*in_c_ + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;*/

        // update w
        Halide::Var n;
        Halide::Func desc_w_f;
        Halide::Buffer<float> dw_view(dw_.get(), size_);
        Halide::Buffer<float> train_w_view(train_w_.get(), size_);
        desc_w_f(n) = train_w_view(n) - lr * dw_view(n) - lambda_ * train_w_view(n);
        /*
        std::cout << "W after: " << std::endl;
        std::cout << std::endl;
        for (int i = 0; i < out_c_; ++i) {
            for (int j = 0; j < k_*k_*in_c_; ++j) {
                std::cout << train_w_[i*k_*k_*in_c_ + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
        */
        desc_w_f.realize(train_w_view);
    }

    void ConvLayer::load_weights(float* real_weights) {
        if (!train_w_) {
            train_w_.reset(new float[size_]);
        }
        for (int i = 0; i < size_; ++i) train_w_[i] = real_weights[i];
    }

    float* ConvLayer::get_dw() {
        return dw_.get();
    }

    float ConvLayer::get_train_w(int x, int y, int in_c, int out_c) {
        return train_w_[out_c * k_ * k_ * in_c_
                                + in_c * k_ * k_
                                + y * k_
                                + x];
    }

    // Helper im2col, same as BatchIm2Col without n
    void ConvIm2Col(Halide::Buffer<float> out, Halide::Buffer<float> in,
                        const int p, const int s, const int k,
                        const int out_width, const int out_height) {
        // in Halide dims: cols, rows, channels, batch
        // out Halide dims: n_patches, kkc, batch
        assert(out.dim(0).extent() == out_width * out_height);
        assert(out.dim(1).extent() == k*k*in.dim(2).extent());
        const int patch_area = k * k;
        // Algo
        Halide::Var x, y; // i is y, j is x
        Halide::Func im2col_f;
        Halide::Expr c = y / patch_area;
        Halide::Expr pix_index_in_patch = y % patch_area;
        Halide::Expr which_row = x / out_width;
        Halide::Expr which_patch_in_row = x % out_width; // x % out_width
        Halide::Expr top_left_y_index = which_row * s - p;
        Halide::Expr top_left_x_index = which_patch_in_row * s - p;
        Halide::Expr row_in_nb = pix_index_in_patch / k;
        Halide::Expr y_index = top_left_y_index + row_in_nb;
        Halide::Expr col_in_nb = pix_index_in_patch % k; // pix_index_in_patch % k
        Halide::Expr x_index = top_left_x_index + col_in_nb;
        /*
        Halide::Expr oob = y_index < 0 || y_index >= in.dim(1).extent() || x_index < 0 || x_index >= in.dim(0).extent();
        im2col_f(x, y, n) = Halide::select(oob, 0.0f, in(x_index, y_index, c, n));*/
        // no oob cos we're doing valid padding only
        im2col_f(x, y) = in(x_index, y_index, c);
        // Schedule
        im2col_f.realize(out);
    }
}