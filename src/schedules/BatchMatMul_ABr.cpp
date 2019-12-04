#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// We will define a generator to auto-schedule.
class AutoScheduled : public Halide::Generator<AutoScheduled> {
public:
    Output<Buffer<float>>  out{"out", 3};
    Input<Buffer<float>>   A{"A", 2};
    Input<Buffer<float>>   B{"B", 3};

    void generate() {
        Halide::Expr k_size = A.dim(0).extent();
        Halide::RDom k(0, k_size);
        batch_matmul(x, y, n) = 0.0f;
        batch_matmul(x, y, n) += A(k, y) * B(x, k, n);
        out(x, y, n) = batch_matmul(x, y, n);
    }

    void schedule() {
        if (auto_schedule) {
            // To provide estimates (min and extent values) for each dimension
            // of pipeline outputs, we use the estimate() method. estimate()
            // takes in (dim_name, min, extent) as arguments.
            out.estimate(x, 0, 1)
                .estimate(y, 0, 1)
                .estimate(n, 0, 100);
            // To provide estimates (min and extent values) for each dimension
            // of the input images ('input', 'filter', and 'bias'), we use the
            // set_bounds_estimate() method. set_bounds_estimate() takes in
            // (min, extent) of the corresponding dimension as arguments.
            A.dim(0).set_bounds_estimate(0, 10); // for CIFAR
            A.dim(1).set_bounds_estimate(0, 10);

            B.dim(0).set_bounds_estimate(0, 10); // for CIFAR
            B.dim(1).set_bounds_estimate(0, 10);
            B.dim(2).set_bounds_estimate(0, 100);
        } else {
            batch_matmul.compute_root();
        }
    }
private:
    Var x{"x"}, y{"y"}, n{"n"};
    Halide::Func batch_matmul;
};

HALIDE_REGISTER_GENERATOR(AutoScheduled, batchmatmulabr_schedule_gen)