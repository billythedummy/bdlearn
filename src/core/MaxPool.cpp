#include "bdlearn/MaxPool.hpp"

namespace bdlearn {

    void MaxPool::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // check dimens
        assert(out.dim(2).extent() == in.dim(2).extent()); // c == c
        assert(out.dim(3).extent() == in.dim(3).extent()); // batch == batch

        Halide::Func f, max_f;
        Halide::Var x, y, dx, dy, c , n;
        Halide::Expr x_ = x * s_;
        Halide::Expr y_ = y * s_;

        f(x, y, dx, dy, c, n) = in(x_ + dx, y_ + dy, c, n);

        Halide::RDom space(0, k_, 0, k_);
        max_f(x, y, c, n) = Halide::argmax(f(x, y, space.x, space.y, c, n));
        Halide::Realization maxes = max_f.realize(out.dim(0).extent(), out.dim(1).extent(), out.dim(2).extent(), out.dim(3).extent());
        max_x_ = maxes[0];
        max_y_ = maxes[1];
        out.copy_from(maxes[2]);
    }

    void MaxPool::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
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
        out.copy_from(maxes[2]);
    }

    void MaxPool::backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) {
        Halide::Var x, y, c, n;
        Halide::Func grad_f;
        Halide::Expr max_x_i = x/k_;
        Halide::Expr max_y_i = y/k_;
        grad_f(x, y, c, n) = Halide::select(
            (x-k_*max_x_i) == max_x_(max_x_i, max_y_i, c, n)
            && (y-k_*max_y_i) == max_y_(max_x_i, max_y_i, c, n), 
            ppg(max_x_i, max_y_i, c, n),
            0.0f
        );
        grad_f.realize(out);
    }

    bufdims MaxPool::calc_out_dim(bufdims in_dims) {
        return {(in_dims.w - k_) / s_ + 1, (in_dims.h - k_) / s_ + 1, in_dims.c};
    }

    void MaxPool::update(float lr) {}
}