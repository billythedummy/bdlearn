#include <iostream>
#include "BMat_basic.hpp"
#include "Halide_test.hpp"
#include "BConv_basic.hpp"

using namespace bdlearn;

int main(int argc, char **argv) {
    if (test_BMat_copy_and_equals()) return -1;
    if (test_BMat_matmul_simple()) return -1;
    if (test_Halide_main()) return -1;
    if (test_BMat_sign_constructor()) return -1;
    if (test_BConv_rand_constructor()) return -1;
    if (test_forward_t()) return -1;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}