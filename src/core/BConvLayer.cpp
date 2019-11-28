#include "bdlearn/BConvLayer.hpp"

namespace bdlearn {
    // Constructors

    // Destructor

    BConvLayer::~BConvLayer() {}

    // public functions

    void BConvLayer::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // TO-DO
        int cols = in.dim(0).extent();
        int rows = in.dim(1).extent();
        const int out_height = (rows - k_) / 1 + 1;
        const int out_width = (cols - k_) / 1 + 1;
        const int patch_area = k_* k_;
        const int h_im2col = patch_area * in_c_;
        const int w_im2col = out_height * out_width;
        float* in_im2col = BConvLayer::im2col<float>(in, rows, cols, in_c_, 0, 0, s_, s_, k_, k_);
        BMat mat_in(h_im2col, w_im2col, in_im2col);
        delete in_im2col;
        std::cout << "im2col" << mat_in.rows() << " " << mat_in.cols() << std::endl;
        matmul(out, w_, mat_in);
        return;
    }

    void BConvLayer::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // TO-DO
        return;
    }

    void BConvLayer::backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) {
        // TO-DO
        return;
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