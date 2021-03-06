#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// We will define a generator to auto-schedule.
class AutoScheduled : public Halide::Generator<AutoScheduled> {
public:
    Output<Buffer<float>>  out{"out", 3};
    Input<Buffer<float>>   in{"in", 4};
    Input<int> p{"p"};
    Input<int> s{"s"};
    Input<int> k{"k"};
    Input<int> out_width{"out_width"};
    Input<int> out_height{"out_height"};

    void generate() {
        Halide::Expr patch_area = k * k;
        // Algo
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
        im2col_f(x, y, n) = in(x_index, y_index, c, n);
        out(x, y, n) = im2col_f(x, y, n);
    }

    void schedule() {
        if (auto_schedule) {
            // To provide estimates (min and extent values) for each dimension
            // of pipeline outputs, we use the estimate() method. estimate()
            // takes in (dim_name, min, extent) as arguments.
            out.estimate(x, 0, 1)
                .estimate(y, 0, 27)
                .estimate(n, 0, 100);
            // To provide estimates (min and extent values) for each dimension
            // of the input images ('input', 'filter', and 'bias'), we use the
            // set_bounds_estimate() method. set_bounds_estimate() takes in
            // (min, extent) of the corresponding dimension as arguments.
            in.dim(0).set_bounds_estimate(0, 1); // for CIFAR
            in.dim(1).set_bounds_estimate(0, 1);
            in.dim(2).set_bounds_estimate(0, 3);
            in.dim(3).set_bounds_estimate(0, 100);
            // To provide estimates on the parameter values, we use the
            // set_estimate() method.
            p.set_estimate(0);
            s.set_estimate(1);
            k.set_estimate(3);
            out_width.set_estimate(28);
            out_height.set_estimate(28);

        } else {
            im2col_f.compute_root();
        }
    }
private:
    Var x{"x"}, y{"y"}, n{"n"};
    Func im2col_f;
};

HALIDE_REGISTER_GENERATOR(AutoScheduled, batchim2col_schedule_gen)