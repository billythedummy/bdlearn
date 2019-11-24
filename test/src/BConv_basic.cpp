#include <iostream>
#include "BConv_basic.hpp"

using namespace bdlearn;

int test_BConv_rand_constructor() {
    size_t k = 13;
    size_t in_c = 3;
    size_t out_c = 5;
    BConvLayer dut (k, 1, in_c, out_c);
    for (size_t out_c_i = 0; out_c_i < out_c; ++out_c_i) {
        for (size_t in_c_i = 0; in_c_i < in_c; ++in_c_i) {
            for (size_t y = 0; y < k; ++y) {
                for (size_t x = 0; x < k; ++x) {
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
