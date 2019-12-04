#include "DataSet_test.hpp"

using namespace bdlearn;

int test_DataSet() {
    const int batch_size = 5;
    DataSet dut;
    dut.load_darknet_classification("/home/dhy1996/uwimg/cifar_mini.test",
                                "/home/dhy1996/uwimg/cifar.labels");
    dut.set_batch_size(batch_size);
    const int classes = dut.get_classes();
    const int x_size = dut.get_x_size();
    std::cout << "Loading data done" << std::endl;
    dut.shuffle();
    std::cout << "Shuffling done" << std::endl;
    batchdata next = dut.get_next_batch();
    bufdims indim = dut.get_x_dims();
    for (int n = 0; n < batch_size; ++n) {
        std::cout << std::endl;
        std::cout << "#" << n << std::endl;
        for (int c = 0; c < indim.c; ++c) {
            for (int y = 0; y <indim.h; ++y) {
                for (int x = 0; x < indim.w; ++x) {
                    std::cout << next.x_ptr[n*x_size
                        + c * indim.w*indim.h
                        + y * indim.w
                        + x] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "Y: " << std::endl;
        for (int k = 0; k < classes; ++k) {
            std:: cout << next.y_ptr[n*classes + k] << ", ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    free_batch_data(next);
    /*
    next = dut.get_next_batch();
    std::cout << next.size << std::endl;
    free_batch_data(next);
    next = dut.get_next_batch();
    std::cout << next.size << std::endl;
    free_batch_data(next);
    next = dut.get_next_batch();
    std::cout << next.size << std::endl;
    free_batch_data(next);*/
    return 0;
}