#include <bdlearn/bdlearn.hpp>
#include <iostream>
#include "BMat_basic.hpp"

using namespace bdlearn;

int main(int argc, char **argv) {
    if (test_BMat_copy_and_equals())
        return -1;
    if (test_BMat_matmul_simple())
        return -1;
    return 0;
}