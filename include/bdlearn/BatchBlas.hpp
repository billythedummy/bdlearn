#ifndef _BDLEARN_BATCHBLAS_H_
#define _BDLEARN_BATCHBLAS_H_

#include "Halide.h"

namespace bdlearn {
    void BatchMatMul(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> B);
}

#endif