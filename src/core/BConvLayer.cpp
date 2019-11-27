#include "bdlearn/BConvLayer.hpp"

namespace bdlearn {
    // Constructors

    // Destructor

    BConvLayer::~BConvLayer() {}

    // public functions

    void BConvLayer::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // TO-DO
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