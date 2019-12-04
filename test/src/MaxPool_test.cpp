#include "MaxPool_test.hpp"
int test_forward_t() {
    const int w = 4;
    const int h = 4;
    const int c = 3;
    const int batch = 5;
    const int X_size = w*h*c*batch;
    const int Y_size = c*batch;

    BMaxPool t1(2, 2);
    float X [X_size];
    for (int i = 0; i < c * batch; i++) {
        for (int j = 0; j < w * h; j++) {
            if (j % 4 == 0) {
                x[i*j] = 10;
            } else {
                X[i*j] = j;
            }
        }
    }

    float Y [Y_size];
    Halide::Buffer<float> X_view(X, w, h, c, batch);
    Halide::Buffer<float> Y_view(Y, 2, 2, c, batch);
    t1.forward_t(Y_view, X_view);

    for (i = 0; i < Y_size; i++) {
        if (Y[i] != 10) {
            return -1;
        }
    }

    return 0;
}

int test_backward_t() {
    return -1;
}