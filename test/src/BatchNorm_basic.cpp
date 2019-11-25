#include <iostream>
#include <random>
#include <chrono>
#include "BatchNorm_basic.hpp"

using namespace bdlearn;

int test_BatchNorm_forward_i() {
    int c = 7;
    BatchNorm dut(c);
    int m = 3;
    int n = 5;
    // random init test array
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / n));
    float test[c*m*n];
    for (int i = 0; i < c*m*n; ++i) test[i] = dist(generator);
    // test if output same with default
    Halide::Buffer<float> test_view(test, n, m, c);
    float out[c*m*n];
    Halide::Buffer<float> out_view(out, n, m, c);
    dut.forward_i(&out_view, test_view);
    for (int i = 0; i < c*m*n; ++i) {
        if (test[i] != out[i]) {
            std::cerr << "BatchNorm simple forward failed at " << i;
            std::cerr << " Expected: " << test[i] << ", got: " << out[i] << std::endl;
            return -1;
        }
    }
    return 0;
}

int test_BatchNorm_forward_t() {
    return 0;
}