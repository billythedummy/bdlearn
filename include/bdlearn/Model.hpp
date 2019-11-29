#ifndef _BDLEARN_MODEL_H_
#define _BDLEARN_MODEL_H_

#include "Halide.h"
#include "bdlearn/Layer.hpp"

namespace bdlearn {
    class Model {
        public:
        // constructor
            Model() {}

        // Destructor
            virtual ~Model();

        // public functions
            void train_step(Halide::Buffer<float> X, Halide::Buffer<float> Y);
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in); // inference
        
        // friend operators
        //friend std::ostream& operator<<(std::ostream& os, const Layer& l);

        private:
            // delete assigment
            Model& operator=(const Model& ref) = delete;
            // private functions
            void backward(Halide::Buffer<float> out, Halide::Buffer<float> loss);
            void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in); // training
            // private fields
            std::unique_ptr<std::unique_ptr<Layer>[]> layer_ptrs_;
            int n_layers_;
            int batch_size_;
            std::unique_ptr<float[]> buf_i_;
            std::unique_ptr<float[]> buf_t_;
    };
}

#endif