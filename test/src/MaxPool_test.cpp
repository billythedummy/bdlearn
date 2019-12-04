#include "MaxPool_test.hpp"
using namespace bdlearn;

const int w = 4;
const int h = 4;
const int c = 3;
const int batch = 5;
const int max = 17;
const int X_size = w*h*c*batch;
const int Y_size = c*batch;
MaxPool pool_test(2, 2);
float X [X_size];
float Y [Y_size];
Halide::Buffer<float> X_view(X, w, h, c, batch);
Halide::Buffer<float> Y_view(Y, 2, 2, c, batch);

int test_MaxPool_forward_t() {
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
    pool_test.forward_t(Y_view, X_view);

    for (int i = 0; i < Y_size; i++) {
        if (Y[i] != max) {
            std::cout << "MaxPool forward_t failed: expected " << max << " but got " << Y[i] << std::endl;
            return -1;
        }
    }

    return 0;
}

int test_MaxPool_backward() {
    pool_test.backward(X_view, Y_view);
    for (int i = 0; i < c * batch; i++) {
        for (int j = 0; j < w * h; j++) {
            if (j % 2 == 0 && X[i] != max) {
                std::cout << "MaxPool backward_t failed: expected " << max << " but got " << X[i] << std::endl;
                return -1;
            } else if (X[i] != 0) {
                std::cout << "MaxPool backward_t failed: expected 0 but got " << X[i] << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

int test_MaxPool() {
    int ret;
    ret += test_MaxPool_backward();
    ret += test_MaxPool_forward_t();
    return ret;
}