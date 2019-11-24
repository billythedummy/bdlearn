#include <iostream>
#include "BConv_basic.hpp"

using namespace bdlearn;

int test_BConv_rand_constructor() {
    int k = 13;
    int in_c = 3;
    int out_c = 5;
    BConvLayer dut (k, 1, in_c, out_c);
    for (int out_c_i = 0; out_c_i < out_c; ++out_c_i) {
        for (int in_c_i = 0; in_c_i < in_c; ++in_c_i) {
            for (int y = 0; y < k; ++y) {
                for (int x = 0; x < k; ++x) {
                    if (dut.get_w(x, y, in_c_i, out_c_i)
                        != (dut.get_train_w(x, y, in_c_i, out_c_i) >= 0)) {
                        std::cerr << "test_BConv_rand_constructor failed at " << out_c_i << ", " << in_c_i << ", " << y << ", " << x;
                        std::cerr << ". Expected: " << (dut.get_train_w(x, y, in_c_i, out_c_i) >= 0) << ", got: " << dut.get_w(x, y, in_c_i, out_c_i) << std::endl;
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

int test_forward_t() {
    int k = 13;
    int in_c = 3;
    int out_c = 5;
    BConvLayer dut (k, 1, in_c, out_c);

    int m = 120; 
    int n = 100;
    float* res = new float[m*n*in_c];
    Halide::Buffer<float> res_buf(res, m, n, "res_buf");

    float out[((n - k) + 1) * ((m - k) + 1)];
    Halide::Buffer<float> out_buf(res, ((m - k) + 1), ((n - k) + 1), "out_buf");
    printf("apple %d %d %d %d",
        res_buf.dim(0).extent(),
        res_buf.dim(1).extent(),
        out_buf.dim(0).extent(),
        out_buf.dim(1).extent()
    )
    dut.forward_t(&out_buf, res_buf);
    delete res;
    return 0;
}