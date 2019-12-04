#include "Model_test.hpp"

using namespace bdlearn;

int test_Model() {
    // train for OR
    const int in_w = 1;
    const int in_h = 1;
    const int in_c = 2;
    const int batch = 10;
    const int classes = 2;
    // define model
    const bufdims in_dims {in_w, in_h, in_c};
    const int X_size = in_w * in_h * in_c * batch;
    Model dut(in_dims, true);
    //dut.append_batch_norm();
    dut.append_bconv(1, classes);
    dut.loss_softmax_cross_entropy();
    // make fake data
    float X [X_size];
    float Y [classes*batch];
    for (int i = 0; i < batch; ++i) {
        bool x_i = i % 2;
        bool x_ip1 = false;
        X[i*in_w*in_h*in_c] = x_i;
        X[i*in_w*in_h*in_c + 1] = x_ip1;
        if (x_i | x_ip1) {
            Y[i*classes] = 1;
            Y[i*classes+1] = 0;
        } else {
            Y[i*classes] = 0;
            Y[i*classes+1] = 1;
        }
    }

    std::cout << "X: " << std::endl;
    for( int i = 0; i < batch; ++i) {
        for (int j = 0; j < in_c; ++j) {
            std::cout << X[i*in_w*in_w*in_c + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Y: " << std::endl;
    for( int i = 0; i < batch; ++i) {
        for (int j = 0; j < classes; ++j) {
            std::cout << Y[i*classes + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    float output [classes*batch];
    float eval_out [batch];
    // test full
    Halide::Buffer<float> X_view(X, in_w, in_h, in_c, batch);
    Halide::Buffer<float> Y_view(Y, classes, batch);

    std::cout << "Initial prediction: " << std::endl;
    dut.forward_batch(output, X_view);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < classes; ++j) {
            std::cout << output[i*classes + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dut.eval(X_view, Y_view, eval_out);
    std::cout << "is wrong: ";
    for (int i = 0; i < batch; ++i) {
        std::cout << eval_out[i] << ", ";
    }
    std::cout << std::endl;

    dut.set_lr(5.0f);
    for (int i = 0; i < 15; ++i) {
        float loss = dut.train_step(X_view, Y_view);
        std::cout << "Loss: " << loss << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Final prediction: " << std::endl;
    dut.forward_batch(output, X_view);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < classes; ++j) {
            std::cout << output[i*classes + j] << ", ";
        }
        std::cout << std::endl;
    }
    dut.eval(X_view, Y_view, eval_out);
    std::cout << "is wrong: ";
    for (int i = 0; i < batch; ++i) {
        std::cout << eval_out[i] << ", ";
    }
    std::cout << std::endl;
    return 0;
}

int test_save_load_model() {
    // train for OR
    const int in_w = 1;
    const int in_h = 1;
    const int in_c = 2;
    const int batch = 10;
    const int classes = 2;
    // define model
    const bufdims in_dims {in_w, in_h, in_c};
    const int X_size = in_w * in_h * in_c * batch;
    Model dut(in_dims, true);
    dut.append_batch_norm();
    dut.append_bconv(1, classes);
    
    std::string path = "./test_weights/ModelTest.csv";
    std::ofstream fout;
    fout.open(path, std::ios::out | std::ios::trunc);
    if (fout.fail()) {
        std::cerr << "File failed to open" << std::endl;
        return -1;
    }
    dut.save_model(fout);
    return 0;
}