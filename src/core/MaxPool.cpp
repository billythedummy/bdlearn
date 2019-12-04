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
        float* buf = out.begin();

        // sets the buffer to be full of 0's
        size_t buf_size = out.dim(0).extent() * out.dim(1).extent() * out.dim(2).extent() * out.dim(3).extent() * sizeof(float);
        memset(buf, 0, buf_size);
        
        int* max_x_arr = max_x_.begin();
        int* max_y_arr = max_y_.begin();
        float* ppg_arr = ppg.begin();
        size_t ppg_w = ppg.dim(0).extent();
        size_t ppg_h = ppg.dim(1).extent();
        size_t ppg_c = ppg.dim(2).extent();
        size_t ppg_n = ppg.dim(3).extent();

        size_t out_w = out.dim(0).extent();
        size_t out_h = out.dim(1).extent();

        size_t cur_x = 0;
        size_t cur_y = 0;
        size_t cur_index = 0;
        for (int n = 0; n < ppg_n; n++) {
            for (int c = 0; c < ppg_c; c++) {
                for (int h = 0; h < ppg_h; h++) {
                    for (int w = 0; w < ppg_w; w++) {
                        cur_index = w + ppg_w * h + ppg_w * ppg_h * c + ppg_w * ppg_h * ppg_c * n;
                        cur_x = (max_x_arr[cur_index] - 1) * s_ + k_;
                        cur_y = (max_y_arr[cur_index] - 1) * s_ + k_;
                        buf[cur_x + out_w * cur_y + out_w * out_h * c + out_w * out_h * ppg_c * n] = ppg_arr[cur_index];
                    }
                }
            }
        } 
    }

    bufdims MaxPool::calc_out_dim(bufdims in_dims) {
        return {(in_dims.w - k_) / s_ + 1, (in_dims.h - k_) / s_ + 1, in_dims.c};
    }

    void MaxPool::update(float lr) {}
}