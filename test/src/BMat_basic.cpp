#include <iostream>
#include "BMat_basic.hpp"

using namespace bdlearn;

int test_BMat_copy_and_equals() {
    BMat orig(2, 4);
    orig.ones();
    BMat copy(orig);
    std::cout << orig << std::endl;
    if (!(orig == copy)) {
        std::cerr << "BMat_copy_and_equals failed!" << std::endl;
        return 1;
    }
    return 0; 
}

int test_BMat_matmul_simple() {
    size_t m = 3; 
    size_t k = 7;
    size_t n = 5;
    BMat s1(m, k);
    BMat s2(k, n);
    s1.random();
    s2.random();
    std::cout << s1 << std::endl;
    std::cout << s2 << std::endl;
    float res[m*n];
    Halide::Buffer<float> res_buf(res, n, m, "res_buf");
    matmul(&res_buf, s1, s2);
    float* disp = res;
    
    /*
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << *disp << " ";
            ++disp;
        }
        std::cout << std::endl;
    }
    */

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0;
            for (size_t x = 0; x < k; x++) {
                uint8_t t1 = s1.get(i, x);
                uint8_t t2 = s2.get(x, j);
                sum += (t1 ? 1 : -1) * (t2 ? 1 : -1);
            }
            std::cout << i << j << sum << *disp << std::endl;
            assert(*disp == sum);
            ++disp;
        }
    }
    return 0;
}


