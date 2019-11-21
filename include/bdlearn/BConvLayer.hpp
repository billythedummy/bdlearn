#ifndef _BDLEARN_BCONVLAYER_H_
#define _BDLEARN_BCONVLAYER_H_

#include <cstddef>
#include <iostream>
#include <memory>
#include "BMat.hpp"
#include "Layer.hpp"

namespace bdlearn {
    class BConvLayer: public Layer {
        public:
        // Constructors
            // default - random initialized
            BConvLayer(size_t k, size_t s, size_t in_c, size_t out_c); // VALID PADDING ONLY

        // Destructor
            virtual ~BConvLayer();

        // public functions
            void forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float>* out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float>* out, Halide::Buffer<float> ppg) override;
            void load_weights();
        
        // friend operators
        friend std::ostream& operator<<(std::ostream& os, const BConvLayer& l);


        private:
            std::unique_ptr<float[]> train_w_;
            size_t k;
            size_t s;
            size_t in_c;
            size_t out_c;
            BMat w_;
            
            BConvLayer& operator=(const BConvLayer& ref) = delete;
    };
}

#endif