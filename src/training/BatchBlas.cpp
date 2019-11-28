#include "bdlearn/BatchBlas.hpp"

namespace bdlearn {

    void BatchMatMul(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> B) {
        // A Halide dims: cols, rows, batch
        // B Halide dims: cols, rows, batch
        // C Halide dims: cols, row, batch
        // A - m rows k cols, B - k rows n cols, C - m rows n cols
        /*
        assert(A.dim(0).extent() == B.dim(1).extent());
        assert(A.dim(1).extent() == out.dim(1).extent());
        assert(B.dim(0).extent() == out.dim(0).extent());
        assert(A.dim(2).extent() == out.dim(2).extent());
        assert(B.dim(2).extent() == out.dim(2).extent());*/
        // Algo
        int k_size = A.dim(0).extent();
        Halide::Var x("x"), xi("xi"), xo("xo"), y("y"), yo("yo"), yi("yi"), yii("yii"), xii("xii"), n("n");
        Halide::RDom k(0, k_size);
        Halide::Func batch_matmul("batch_matmul");
        batch_matmul(x, y, n) = 0.0f;
        batch_matmul(x, y, n) += A(k, y, n) * B(x, k, n);
        Halide::Func out_f;
        out_f(x, y, n) = batch_matmul(x, y, n);
        // Schedule
        //Halide::Var xy;
        // not smart enough to optimize this
        out_f.realize(out);
    }

}