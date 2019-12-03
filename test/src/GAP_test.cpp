#include "GAP_test.hpp"

using namespace bdlearn;

int test_GAP_forward_backward_t() {
    const int w = 4;
    const int h = 3;
    const int c = 2;
    const int batch = 5;
    const int X_size = w*h*c*batch;
    const int Y_size = c*batch;
    GAP dut(true);
    float X [X_size];
    for (int i = 0; i < X_size; ++i) {
        X[i] = rand() / 10000000;
    }
    float Y [Y_size];
    Halide::Buffer<float> X_view(X, w, h, c, batch);
    Halide::Buffer<float> Y_view(Y, 1, 1, c, batch);
    dut.forward_t(Y_view, X_view);

    for (int i = 0; i < Y_size; ++i) {
        float ave = 0.0f;
        int offset = i * w * h;
        for (int j = 0; j < w*h; ++j) {
            ave += X[offset + j];
        }
        ave /= w*h;
        if (fabsf(ave - Y[i]) > 1E-3f) {
            std::cerr << "GAP test failed forward at " << i;
            std::cerr << ". Expected: " << ave << ". Got : " << Y[i] << std::endl;
            std::cerr << fabsf(ave - Y[i]);
            return -1;
        }
    }

    dut.backward(X_view, Y_view); // note: this overwrites X
    for (int i = 0; i < X_size; ++i) {
        int which_cb = i / (w*h);
        if (fabsf(X[i] - Y[which_cb] / (w*h)) > 1E-3f) {
            std::cerr << "GAP test failed backward at " << i;
            std::cerr << ". Expected: " << Y[which_cb] / (w*h) << ". Got : " << X[i] << std::endl;
            return -1;
        }
    }
    return 0;
}
