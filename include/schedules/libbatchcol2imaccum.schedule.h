#ifndef libbatchcol2imaccum_SCHEDULE_H
#define libbatchcol2imaccum_SCHEDULE_H

// MACHINE GENERATED -- DO NOT EDIT
// This schedule was automatically generated by src/AutoSchedule
// for target=x86-64-linux-avx-avx2-f16c-fma-sse41  // NOLINT
// with machine_params=32,16777216,40

#include "Halide.h"


inline void apply_schedule_libbatchcol2imaccum(
    ::Halide::Pipeline pipeline,
    ::Halide::Target target
) {
    using ::Halide::Func;
    using ::Halide::MemoryType;
    using ::Halide::RVar;
    using ::Halide::TailStrategy;
    using ::Halide::Var;
    Var n_vi("n_vi");
    Var n_vo("n_vo");

    Func col2im_accum_f_1 = pipeline.get_func(1);
    Func out = pipeline.get_func(2);

    {
        Var n = col2im_accum_f_1.args()[3];
        col2im_accum_f_1
            .compute_root()
            .split(n, n_vo, n_vi, 8)
            .vectorize(n_vi)
            .parallel(n_vo);
        col2im_accum_f_1.update(0)
            .split(n, n_vo, n_vi, 8, TailStrategy::GuardWithIf)
            .vectorize(n_vi)
            .parallel(n_vo);
    }
    {
        Var n = out.args()[3];
        out
            .compute_root()
            .split(n, n_vo, n_vi, 8)
            .vectorize(n_vi)
            .parallel(n_vo);
    }


}

#endif  // libbatchcol2imaccum_SCHEDULE_H
