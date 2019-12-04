#ifndef libbatchim2col_SCHEDULE_H
#define libbatchim2col_SCHEDULE_H

// MACHINE GENERATED -- DO NOT EDIT
// This schedule was automatically generated by src/AutoSchedule
// for target=x86-64-linux-avx-avx2-f16c-fma-sse41  // NOLINT
// with machine_params=32,16777216,40

#include "Halide.h"


inline void apply_schedule_libbatchim2col(
    ::Halide::Pipeline pipeline,
    ::Halide::Target target
) {
    using ::Halide::Func;
    using ::Halide::MemoryType;
    using ::Halide::RVar;
    using ::Halide::TailStrategy;
    using ::Halide::Var;
    Var y_vi("y_vi");
    Var y_vo("y_vo");

    Func out = pipeline.get_func(2);

    {
        Var y = out.args()[1];
        Var n = out.args()[2];
        out
            .compute_root()
            .split(y, y_vo, y_vi, 8)
            .vectorize(y_vi)
            .parallel(n);
    }


}

#endif  // libbatchim2col_SCHEDULE_H
