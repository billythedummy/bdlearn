#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// We will define a generator to auto-schedule.
class AutoScheduled : public Halide::Generator<AutoScheduled> {
public:
    Output<Buffer<float>>  out{"out", 4};
    Input<Buffer<float>>   in{"in", 3};
    Input<int> p{"p"};
    Input<int> s{"s"};
    Input<int> k{"k"};
    Input<int> out_width{"out_width"};
    Input<int> out_height{"out_height"};

    void generate() {
        Halide::Expr patch_area = k * k;
        // Algo
        Halide::RDom nb(0, k, 0, k);
        Halide::Func col2im_accum_f;
        col2im_accum_f(x, y, c, n) = 0.0f;
        Halide::Expr row_index = c * patch_area + nb.x + nb.y*k;
        Halide::Expr top_left_y = y - nb.y;
        Halide::Expr top_left_x = x - nb.x;
        Halide::Expr which_patch_row = (top_left_y + p) / s;
        Halide::Expr which_patch_in_row = (top_left_x + p) / s;
        Halide::Expr which_patch = which_patch_row * out_width + which_patch_in_row; 
        Halide::Expr invalid = ((top_left_y + p) % s != 0) || ((top_left_x + p) % s != 0)
                                || which_patch_in_row < 0 || which_patch_in_row >= out_width
                                || which_patch_row < 0 || which_patch_row >= out_height;
        Halide::Expr which_patch_clamped = Halide::clamp(which_patch, 0, out_width * out_height - 1);
        col2im_accum_f(x, y, c, n) += Halide::select(invalid, 0.0f, in(which_patch_clamped, row_index, n));
        out(x, y, c, n) = col2im_accum_f(x, y, c, n);
    }

    void schedule() {
        if (auto_schedule) {
            // To provide estimates (min and extent values) for each dimension
            // of pipeline outputs, we use the estimate() method. estimate()
            // takes in (dim_name, min, extent) as arguments.
            out.estimate(x, 0, 1)
                .estimate(y, 0, 1)
                .estimate(c, 0, 3)
                .estimate(n, 0, 100);
            // To provide estimates (min and extent values) for each dimension
            // of the input images ('input', 'filter', and 'bias'), we use the
            // set_bounds_estimate() method. set_bounds_estimate() takes in
            // (min, extent) of the corresponding dimension as arguments.
            in.dim(0).set_bounds_estimate(0, 1); // for CIFAR
            in.dim(1).set_bounds_estimate(0, 27);
            in.dim(2).set_bounds_estimate(0, 100);
            // To provide estimates on the parameter values, we use the
            // set_estimate() method.
            p.set_estimate(0);
            s.set_estimate(1);
            k.set_estimate(3);
            out_width.set_estimate(28);
            out_height.set_estimate(28);

        } else {
            col2im_accum_f.compute_root();
        }
    }
private:
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Func col2im_accum_f;
};

HALIDE_REGISTER_GENERATOR(AutoScheduled, batchcol2imaccum_schedule_gen)