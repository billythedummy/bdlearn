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
        Halide::Func batch_matmul("batch_matmul_BT");
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
        Halide::Func batch_matmul("batch_matmul_AT");
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

    void BatchMatMul_ABr(Halide::Buffer<float> out, Halide::Buffer<float> A, Halide::Buffer<float> B) {
        // A Halide dims: cols, rows
        // B Halide dims: cols, rows, batch
        // C Halide dims: cols, row, batch
        // A - m rows k cols, B - k rows n cols, C - m rows n cols
        assert(A.dim(0).extent() == B.dim(1).extent());
        assert(A.dim(1).extent() == out.dim(1).extent());
        assert(B.dim(0).extent() == out.dim(0).extent());
        assert(B.dim(2).extent() == out.dim(2).extent());
        // Algo
        int k_size = A.dim(0).extent();
        Halide::Var x("x"), y("y"), n("n");
        Halide::RDom k(0, k_size);
        Halide::Func batch_matmul("batch_matmul_ABr");
        batch_matmul(x, y, n) = 0.0f;
        batch_matmul(x, y, n) += A(k, y) * B(x, k, n);
        Halide::Func out_f;
        out_f(x, y, n) = batch_matmul(x, y, n);
        // Schedule
        //Halide::Var xy;
        //Halide::Var xi("xi"), xo("xo"), yo("yo"), yi("yi"), yii("yii"), xii("xii"),
        // not smart enough to optimize this
        out_f.realize(out);
    }

    void BatchIm2Col(Halide::Buffer<float> out, Halide::Buffer<float> in,
                        const int p, const int s, const int k,
                        const int out_width, const int out_height) {
        // in Halide dims: cols, rows, channels, batch
        // out Halide dims: n_patches, kkc, batch
        assert(out.dim(0).extent() == out_width * out_height);
        assert(out.dim(1).extent() == k*k*in.dim(2).extent());
        assert(out.dim(2).extent() == in.dim(3).extent());
        const int patch_area = k * k;
        // Algo
        Halide::Var x, y, n; // i is y, j is x
        Halide::Func im2col_f;
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
        // Schedule
        im2col_f.realize(out);
    }
}