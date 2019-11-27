#include <iostream>
#include <random>
#include "BMat_basic.hpp"

using namespace bdlearn;

int test_BMat_copy_and_equals() {
    BMat orig(2, 4);
    orig.ones();
    BMat copy(orig);
    std::cout << orig << std::endl;
    if (!(orig == copy)) {
        std::cerr << "BMat_copy_and_equals failed!" << std::endl;
        return -1;
    }
    return 0; 
}

int test_BMat_matmul_simple() {
    int m = 3; 
    int k = 7;
    int n = 5;
    BMat s1(m, k);
    BMat s2(k, n);
    s1.random();
    s2.random();
    std::cout << s1 << std::endl;
    std::cout << s2 << std::endl;
    float res[m*n];
    Halide::Buffer<float> res_buf(res, n, m, "res_buf");
    matmul(res_buf, s1, s2);
    float* disp = res;
    
    /*
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << *disp << " ";
            ++disp;
        }
        std::cout << std::endl;
    }
    */

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0;
            for (int x = 0; x < k; x++) {
                uint8_t t1 = s1.get(i, x);
                uint8_t t2 = s2.get(x, j);
                sum += (t1 ? 1 : -1) * (t2 ? 1 : -1);
            }
            std::cout << i << j << sum << *disp << std::endl;
            if (*disp != sum) {
                std::cerr << "test_BMat_matmul_simple failed at " << i << ", " << j;
                std::cerr << ". Expected: " << sum << ", got: " << *disp << std::endl;
                return -1;
            }
            ++disp;
        }
    }
    return 0;
}

int test_BMat_sign_constructor() {
    int m = 17;
    int n = 7;
    float rand[m*n];
    for (int i = 0; i < m*n; ++i) rand[i] = std::rand();
    BMat test(m, n, rand);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (rand[i*n + j] >= 0 != test.get(i, j)) {
                std::cerr << "test_BMat_sign_constructor failed at " << i << ", " << j;
                std::cerr << ". Expected: " << (rand[i*n + j] >= 0) << ", got: " << test.get(i, j) << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

