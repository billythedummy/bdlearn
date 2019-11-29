#ifndef _BDLEARN_BATCHBLAS_H_
#define _BDLEARN_BATCHBLAS_H_

#include "Halide.h"

namespace bdlearn {
    void BatchMatMul(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> B);
    void BatchMatMul_BT(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> BT);
    // A broadcasted, transposed
    void BatchMatMul_ATBr(Halide::Buffer<float> out, Halide::Buffer<float> AT, Halide::Buffer<float> B);
    // A broadcasted
    void BatchMatMul_ABr(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> B);
    void BatchIm2Col(Halide::Buffer<float> out, Halide::Buffer<float> in,
                        const int p, const int s, int k, const int out_width, const int out_height);
}

#endif