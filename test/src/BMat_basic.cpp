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
    size_t n = 8;
    size_t k = 8;
    size_t m = 8;
    BMat s1(n, k);
    BMat s2(k, m);
    //std::cout << s1 << std::endl;
    s2.ones();
    //std::cout << s2 << std::endl;
    float res[n*m];
    matmul(res, s1, s2);
    float* disp = res;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            std::cout << *disp << " ";
            ++disp;
        }
        std::cout << std::endl;
    }
    return 0;
}


