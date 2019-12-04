#include "bdlearn/MaxPool.hpp"

namespace bdlearn {

    void BMaxPool::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        prev_in_ = in;
        has_batches = false;
        // check dimens
        assert(out.dim(0).extent() == 1 && out.dim(1).extent() == 1); // w, h == 1
        assert(out.dim(2).extent() == in.dim(2).extent()); // c == c
        assert(out.dim(3).extent() == in.dim(3).extent()); // batch == batch

        Halide::Func f, max_f;
        Halide::Var x, y, dx, dy, c , n;
        Halide::Expr x_ = x * s_;
        Halide::Expr y_ = y * s_;

        f(x, y, dx, dy, c, n) = in(x_ + dx, y_ + dy, c, n);

        Halide::RDom space(0, k_, 0, k_);
        max_f(x, y, c) = Halide::argmax(f(x, y, space.x, space.y, c));
        Halide::Realization maxes = max_f.realize(out.dim(0).extent(), out.dim(1).extent(), out.dim(2).extent(), out.dim(3).extent());
        max_x_ = maxes[0];
        max_y_ = maxes[1];
        max_val_ = maxes[4];
        out.copy_from(max_val_);
    }

    void BMaxPool::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        prev_in_ = in;
        has_batches = true;
        // check dimens
        assert(out.dim(0).extent() == 1 && out.dim(1).extent() == 1); // w, h == 1
        assert(out.dim(2).extent() == in.dim(2).extent()); // c == c

        Halide::Func f, max_f, get_max_indices;
        Halide::Var x, y, dx, dy, c;
        Halide::Expr x_ = x * s_;
        Halide::Expr y_ = y * s_;

        f(x, y, dx, dy, c) = in(x_ + dx, y_ + dy, c);

        Halide::RDom space(0, k_, 0, k_);
        max_f(x, y, c) = Halide::argmax(f(x, y, space.x, space.y, c));
        Halide::Realization maxes = max_f.realize(out.dim(0).extent(), out.dim(1).extent(), out.dim(2).extent());
        max_x_ = maxes[0];
        max_y_ = maxes[1];
        max_val_ = maxes[3];
        out.copy_from(max_val_);
    }

    void BMaxPool::backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) {
        float* buf = out.begin();

        // sets the buffer to be full of 0's
        size_t buf_size = prev_in_.dim(0).extent() * prev_in_.dim(1).extent() * prev_in_.dim(2).extent() * sizeof(float);
        if (has_batches) {
            buf_size *= prev_in_.dim(3).extent();
        }
        memset(buf, 0, buf_size);
        
        int* max_x_arr = max_x_.begin();
        int* max_y_arr = max_y_.begin();
        float* max_val_arr = max_val_.begin();
        for (int i = 0; i < max_x_.dim(0).extent(); i++) {
            buf[max_x_arr[i] + max_x_arr[i] * max_y_arr[i]] = max_val_arr[i]; 
        } 
    }
}