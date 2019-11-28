#ifndef _BDLEARN_BATCHBLAS_H_
#define _BDLEARN_BATCHBLAS_H_

#include "Halide.h"

namespace bdlearn {
    void BatchMatMul(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> B);
    void BatchMatMul_BT(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> BT);
    void BatchMatMul_AT(Halide::Buffer<float> out, Halide::Buffer<float> AT, Halide::Buffer<float> B);
}

#endif