#include "bdlearn/Model.hpp"

namespace bdlearn {
    // destructor
    Model::~Model() {
        buf_i_.reset();
        buf_t_.reset();
        // layer_ptrs_
    }

    // public functions
    void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {

    }
}