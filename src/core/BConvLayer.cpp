#include "bdlearn/BConvLayer.hpp"

namespace bdlearn {
    // Constructors

    // Destructor

    BConvLayer::~BConvLayer() {}

    // public functions

    void BConvLayer::forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in) {
        // TO-DO
        float* in_im2col = BConvLayer::im2col<float>(in, in.dim(0).extent(), in.dim(0).extent(), in_c_, 0, 0, s_, s_, k_, k_);
        
        delete in_im2col;
        return;
    }

    void BConvLayer::forward_i(Halide::Buffer<float>* out, Halide::Buffer<float> in) {
        // TO-DO
        return;
    }

    void BConvLayer::backward(Halide::Buffer<float>* out, Halide::Buffer<float> ppg) {
        // TO-DO
        return;
    }

    uint8_t BConvLayer::get_w(size_t x, size_t y, size_t in_c, size_t out_c) {
        return w_.get(out_c, in_c * k_ * k_ + y * k_ + x);
    }

    float BConvLayer::get_train_w(size_t x, size_t y, size_t in_c, size_t out_c) {
        return train_w_.get()[out_c * k_ * k_ * in_c_
                                + in_c * k_ * k_
                                + y * k_
                                + x];
    }
}