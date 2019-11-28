#include "BatchBlas_test.hpp"

using namespace bdlearn;

int test_BatchMM() {
    int batch = 3;
    int m = 7;
    int n = 5;
    int k = 11;
    float out[batch*m*n];
    float A[batch*m*k];
    float B[batch*k*n];

    for (int i = 0; i < batch*m*k; ++i) {
        A[i] = (float) i;
    }
    for (int i = 0; i < batch*k*n; ++i) {
        B[i] = (float) (i-9);
    }
    Halide::Buffer<float> A_buf(A, k, m, batch);
    Halide::Buffer<float> B_buf(B, n, k, batch);
    Halide::Buffer<float> out_buf(out, n, m, batch);
    BatchMatMul(out_buf, A_buf, B_buf);
    for (int b = 0; b < batch; ++b) {
        for (int y = 0; y < m; ++y) {
            for (int x = 0; x < n; ++x) {
                int out_index = x + y*n + b*m*n;
                float calculated = out[out_index];
                float expected = 0.0f;
                for (int ki = 0; ki < k; ++ki) {
                    int a_index = ki + y*k + b*k*m;
                    int b_index = x + ki*n + b*n*k;
                    expected += A[a_index] * B[b_index];
                }
                if (expected != calculated) {
                    std::cerr << "test_BatchMM failed at " << x << ", " << y << ", " << b;
                    std::cerr << ". Expected: " << expected << ", got: " << calculated << std::endl;
                    return -1;
                }
            }
        }
    }
    return 0;
}