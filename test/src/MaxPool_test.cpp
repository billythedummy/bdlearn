#include "MaxPool_test.hpp"
using namespace bdlearn;

int test_MaxPool_forward_t() {
    const int w = 4;
    const int h = 4;
    const int c = 3;
    const int batch = 5;
    const int max = 17;
    const int X_size = w*h*c*batch;
    const int Y_size = c*batch;

    MaxPool t1(2, 2);
    float X [X_size];
    for (int i = 0; i < c * batch; i++) {
        for (int j = 0; j < w * h; j++) {
            if (j % 2 == 0) {
                X[i*w*h+j] = max;
            } else {
                X[i*w*h+j] = j;
            }
            std::cout << X[i*w*h+j] << ", ";
        }
        std::cout << std::endl;
    }

    float Y [Y_size];
    Halide::Buffer<float> X_view(X, w, h, c, batch);
    Halide::Buffer<float> Y_view(Y, 2, 2, c, batch);
    t1.forward_t(Y_view, X_view);

    for (int i = 0; i < Y_size; i++) {
        if (Y[i] != max) {
            std::cout << Y[i] << std::endl;
            return -1;
        }
    }

    return 0;
}

int test_MaxPool_backward_t() {
    return -1;
}