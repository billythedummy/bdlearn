#include "bdlearn/BatchBlas.hpp"

namespace bdlearn {

    void BatchMatMul(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> B) {
        // A Halide dims: cols, rows, batch
        // B Halide dims: cols, rows, batch
        // C Halide dims: cols, row, batch
        // A - m rows k cols, B - k rows n cols, C - m rows n cols
        assert(A.dim(0).extent() == B.dim(1).extent());
        assert(A.dim(1).extent() == out.dim(1).extent());
        assert(B.dim(0).extent() == out.dim(0).extent());
        assert(A.dim(2).extent() == out.dim(2).extent());
        assert(B.dim(2).extent() == out.dim(2).extent());
        // Algo
        int k_size = A.dim(0).extent();
        Halide::Var x("x"), y("y"), n("n");
        Halide::RDom k(0, k_size);
        Halide::Func batch_matmul("batch_matmul");
        batch_matmul(x, y, n) = 0.0f;
        batch_matmul(x, y, n) += A(k, y, n) * B(x, k, n);
        Halide::Func out_f;
        out_f(x, y, n) = batch_matmul(x, y, n);
        // Schedule
        //Halide::Var xy;
        //Halide::Var xi("xi"), xo("xo"), yo("yo"), yi("yi"), yii("yii"), xii("xii"),
        // not smart enough to optimize this
        out_f.realize(out);
    }

    void BatchMatMul_BT(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> BT) {
        // A Halide dims: cols, rows, batch
        // BT Halide dims: cols, rows, batch
        // C Halide dims: cols, row, batch
        // A - m rows k cols, BT - n rows k cols, C - m rows n cols
        // Each batch output is A @ B.T
        assert(A.dim(0).extent() == BT.dim(0).extent());
        assert(A.dim(1).extent() == out.dim(1).extent());
        assert(BT.dim(1).extent() == out.dim(0).extent());
        assert(A.dim(2).extent() == out.dim(2).extent());
        assert(BT.dim(2).extent() == out.dim(2).extent());
        // Algo
        int k_size = A.dim(0).extent();
        Halide::Var x("x"), y("y"), n("n");
        Halide::RDom k(0, k_size);
        Halide::Func batch_matmul("batch_matmul");
        batch_matmul(x, y, n) = 0.0f;
        batch_matmul(x, y, n) += A(k, y, n) * BT(k, x, n);
        Halide::Func out_f;
        out_f(x, y, n) = batch_matmul(x, y, n);
        // Schedule
        //Halide::Var xy;
        //Halide::Var xi("xi"), xo("xo"), yo("yo"), yi("yi"), yii("yii"), xii("xii"),
        // not smart enough to optimize this
        out_f.realize(out);
    }

    void BatchMatMul_AT(Halide::Buffer<float> out, Halide::Buffer<float> AT, Halide::Buffer<float> B) {
        // AT Halide dims: cols, rows, batch
        // B Halide dims: cols, rows, batch
        // C Halide dims: cols, row, batch
        // A - k rows m cols, BT - k rows n cols, C - m rows n cols
        // Each batch's output is A.T @ B
        assert(AT.dim(1).extent() == B.dim(1).extent());
        assert(AT.dim(0).extent() == out.dim(1).extent());
        assert(B.dim(0).extent() == out.dim(0).extent());
        assert(AT.dim(2).extent() == out.dim(2).extent());
        assert(B.dim(2).extent() == out.dim(2).extent());
        // Algo
        int k_size = AT.dim(1).extent();
        Halide::Var x("x"), y("y"), n("n");
        Halide::RDom k(0, k_size);
        Halide::Func batch_matmul("batch_matmul");
        batch_matmul(x, y, n) = 0.0f;
        batch_matmul(x, y, n) += AT(y, k, n) * B(x, k, n);
        Halide::Func out_f;
        out_f(x, y, n) = batch_matmul(x, y, n);
        // Schedule
        //Halide::Var xy;
        //Halide::Var xi("xi"), xo("xo"), yo("yo"), yi("yi"), yii("yii"), xii("xii"),
        // not smart enough to optimize this
        out_f.realize(out);
    }

}