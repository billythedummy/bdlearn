#include <iostream>
#include <bdlearn/bdlearn.hpp>

#include "BMat_basic.hpp"
#include "Halide_test.hpp"
#include "BConv_basic.hpp"
#include "BatchNorm_basic.hpp"
#include "BatchBlas_test.hpp"
#include "Model_test.hpp"
#include "Softmax_test.hpp"

using namespace bdlearn;

int main(int argc, char **argv) {
    //if (test_BMat_copy_and_equals()) return -1;
    //if (test_BMat_matmul_simple()) return -1;
    //if (test_Halide_main()) return -1;
    //if (test_BMat_sign_constructor()) return -1;
    //if (test_BConv_rand_constructor()) return -1;
    //if (test_BatchNorm_forward_i()) return -1;
    //if (test_BatchNorm_forward_backward_t()) return -1;
    //if (test_forward_t()) return -1;
    //if (test_BatchMM()) return -1;
    //if (test_BatchMMBT()) return -1;
    //if (test_BatchMMATBr()) return -1;
    //if (test_BConv_forward_backward()) return -1;
    //if (test_Model()) return -1;
    if (test_softmax()) return -1;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}