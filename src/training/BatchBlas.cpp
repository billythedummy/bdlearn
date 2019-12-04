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
        // out_f.parallel(n);
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
        // not smart enough to optimize this
        //out_f.parallel(n);
        out_f.realize(out);
    }

    void BatchMatMul_ATBr(Halide::Buffer<float> out, Halide::Buffer<float> AT, Halide::Buffer<float> B) {
        // AT Halide dims: cols, rows
        // B Halide dims: cols, rows, batch
        // C Halide dims: cols, row, batch
        // A - k rows m cols, BT - k rows n cols, C - m rows n cols
        // Each batch's output is A.T @ B
        assert(AT.dim(1).extent() == B.dim(1).extent());
        assert(AT.dim(0).extent() == out.dim(1).extent());
        assert(B.dim(0).extent() == out.dim(0).extent());
        assert(B.dim(2).extent() == out.dim(2).extent());
        // Algo
        int k_size = AT.dim(1).extent();
        Halide::Var x("x"), y("y"), n("n");
        Halide::RDom k(0, k_size);
        Halide::Func batch_matmul("batch_matmul_AT");
        batch_matmul(x, y, n) = 0.0f;
        batch_matmul(x, y, n) += AT(y, k) * B(x, k, n);
        Halide::Func out_f;
        out_f(x, y, n) = batch_matmul(x, y, n);
        // Schedule
        // not smart enough to optimize this
        //Halide::Var xi, yi, xy, yii;
        //out_f.parallel(n);
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
        // not smart enough to optimize this
        //out_f.parallel(n);
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
        
        Halide::Var x_outer, y_outer, x_inner, y_inner, tile_index;
        Halide::Expr tile_dim_x = out.dim(0).extent() > 64 ? 64 : out.dim(0).extent();
        Halide::Expr tile_dim_y = out.dim(1).extent() > 64 ? 64 : out.dim(1).extent();
        im2col_f.tile(x, y, x_outer, y_outer, x_inner, y_inner, tile_dim_x, tile_dim_y)
                .fuse(x_outer, y_outer, tile_index)
                .parallel(tile_index);
        Halide::Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
        Halide::Expr vec_dim_x = out.dim(0).extent() > 4 ? 4 : out.dim(0).extent();
        Halide::Expr pair_dim_y = out.dim(1).extent() > 2 ? 2 : out.dim(1).extent();
        im2col_f.tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, vec_dim_x, pair_dim_y)
                .vectorize(x_vectors)
                .unroll(y_pairs);
                
        im2col_f.realize(out);
    }

    void BatchCol2ImAccum(Halide::Buffer<float> out, Halide::Buffer<float> in,
                            const int p, const int s, const int k,
                            const int out_width, const int out_height) {
        // in Halide dims: n_patches (out_w*out_h), kkinc, batch
        // out Halide dims: in_w, in_h, c_in, batch
        assert(in.dim(0).extent() == out_width * out_height);
        assert(in.dim(1).extent() == k*k*out.dim(2).extent());
        assert(in.dim(2).extent() == out.dim(3).extent());
        const int patch_area = k * k;
        // Algo
        Halide::Var x, y, c, n;
        Halide::RDom nb(0, k, 0, k);
        Halide::Func col2im_accum_f;
        col2im_accum_f(x, y, c, n) = 0.0f;
        Halide::Expr row_index = c * patch_area + nb.x + nb.y*k;
        Halide::Expr top_left_y = y - nb.y;
        Halide::Expr top_left_x = x - nb.x;
        Halide::Expr which_patch_row = (top_left_y + p) / s;
        Halide::Expr which_patch_in_row = (top_left_x + p) / s;
        Halide::Expr which_patch = which_patch_row * out_width + which_patch_in_row; 
        Halide::Expr invalid = ((top_left_y + p) % s != 0) || ((top_left_x + p) % s != 0)
                                || which_patch_in_row < 0 || which_patch_in_row >= out_width
                                || which_patch_row < 0 || which_patch_row >= out_height;
        Halide::Expr which_patch_clamped = Halide::clamp(which_patch, 0, out_width * out_height - 1);
        col2im_accum_f(x, y, c, n) += Halide::select(invalid, 0.0f, in(which_patch_clamped, row_index, n));
        // Schedule
        //col2im_accum_f.parallel(n);
        col2im_accum_f.realize(out);
    }
}