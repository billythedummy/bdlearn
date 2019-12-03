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
        BConvIm2Col(in_im2col_view, in, 0, s_, k_, out_width, out_height);
        // Make BMat with in_im2col
        BMat in_mat(h_im2col, w_im2col, in_im2col);
        // Matmul weights with BMat
        float* out_begin = out.get()->begin(); // this is super hacky i know
        Halide::Buffer<float> out_view(out_begin, in_mat.cols(), w_.rows());
        matmul(out_view, w_, in_mat);

        prev_in_ = in;
        prev_i2c_.reset(in_im2col);
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

    bufdims BConvLayer::calc_out_dim(bufdims in_dims) {
        assert(in_dims.c == in_c_);
        return bufdims {
            (in_dims.w - k_) / s_ + 1,
            (in_dims.h - k_) / s_ + 1,
            out_c_
        };
    }

    void BConvLayer::update(float lr) {
        /*
        std::cout << "W: " << std::endl;
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
        desc_w_f(n) = train_w_view(n) - lr * dw_view(n);
        desc_w_f.realize(train_w_view);
    }
    void BConvLayer::save_layer(std::fstream fout) {
        /*
         * Only save train_w, w_ can be implied from train_w
         */
        for (int i = 0; i < size_; ++i) {
            fout << train_w_.get()[i] << ",";
        }
        fout << "\n";
    }
    void BConvLayer::load_layer(std::fstream fin) {
        std::string line, data;
        getline(fin, line);
        std::istringstream s(line);
        for (int i = 0; i < size_; ++i) {
            std::getline(s, data, ',');
            train_w_.get()[i] = std::stof(data);
        }
        w_.sign(train_w_.get());
    }

    void BConvLayer::load_weights(float* real_weights) {
        if (!train_w_) {
            train_w_.reset(new float[size_]);
        }
        for (int i = 0; i < size_; ++i) train_w_[i] = real_weights[i];
        w_.sign(train_w_.get());
    }

    uint8_t BConvLayer::get_w(int x, int y, int in_c, int out_c) {
        return w_.get(out_c, in_c * k_ * k_ + y * k_ + x);
    }

    float* BConvLayer::get_dw() {
        return dw_.get();
    }

    float BConvLayer::get_train_w(int x, int y, int in_c, int out_c) {
        return train_w_[out_c * k_ * k_ * in_c_
                                + in_c * k_ * k_
                                + y * k_
                                + x];
    }

    // Helper im2col, same as BatchIm2Col without n
    void BConvIm2Col(Halide::Buffer<float> out, Halide::Buffer<float> in,
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