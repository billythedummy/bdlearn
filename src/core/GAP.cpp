#include "bdlearn/GAP.hpp"

namespace bdlearn {
    // public functions
    void GAP::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // dim check
        assert(out.dim(0).extent() == 1 && out.dim(1).extent() == 1); // w, h == 1
        assert(out.dim(2).extent() == in.dim(2).extent()); // c == c
        assert(out.dim(3).extent() == in.dim(3).extent()); // batch == batch
        // mean algo
        Halide::Var x, y, c, n;
        const int w = in.dim(0).extent();
        const int h = in.dim(1).extent();
        Halide::RDom space(0, w, 0, h);
        Halide::Func gap_f;
        gap_f(x, y, c, n) = 0.0f;
        gap_f(x, y, c, n) += in(space.x, space.y, c, n) / (w * h);
        // mean schedule
        gap_f.realize(out);
    }

    void GAP::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // dim check
        assert(out.dim(0).extent() == 1 && out.dim(1).extent() == 1); // w, h == 1
        assert(out.dim(2).extent() == in.dim(2).extent()); // c == c
        // mean algo
        Halide::Var x, y, c;
        const int w = in.dim(0).extent();
        const int h = in.dim(1).extent();
        Halide::RDom space(0, w, 0, h);
        Halide::Func gap_f;
        gap_f(x, y, c) = 0.0f;
        gap_f(x, y, c) += in(space.x, space.y, c) / (w * h);
        // mean schedule
        gap_f.realize(out);
    }

    void GAP::backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) {
        // algo
        Halide::Func dgap_f;
        Halide::Var x, y, c, n;
        dgap_f(x, y, c, n) = ppg(0, 0, c, n) / (out.dim(0).extent() * out.dim(1).extent());
        // schedule
        dgap_f.realize(out);
    }

    bufdims GAP::calc_out_dim(bufdims in_dims) {
        return {1, 1, in_dims.c};
    }

    void GAP::update(float lr) {} // no trainable params
}