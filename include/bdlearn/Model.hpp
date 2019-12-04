#ifndef _BDLEARN_MODEL_H_
#define _BDLEARN_MODEL_H_

#include "Halide.h"
#include "bdlearn/Layer.hpp"
#include "bdlearn/LossLayer.hpp"
#include "bdlearn/SoftmaxCrossEntropy.hpp"
#include "bdlearn/WeightedSoftmaxCrossEntropy.hpp"
#include "bdlearn/BatchNorm.hpp"
#include "bdlearn/BConvLayer.hpp"
#include "bdlearn/GAP.hpp"

namespace bdlearn {
    class Model {
        public:
        // constructor
            Model(const bufdims in_dims, const bool training) 
            : in_dims_(in_dims),
            out_dims_(in_dims),
            training_(training), batch_size_(0),
            lr_(1E-3f) {}

        // Destructor
            virtual ~Model();

        // public functions
            float train_step(Halide::Buffer<float> X, Halide::Buffer<float> Y, void* loss_args=nullptr);
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in); // inference
            void forward_batch(float* out, Halide::Buffer<float> in);
            float eval(Halide::Buffer<float> X, Halide::Buffer<float> Y, void* is_wrong_out);
            void save_model(std::ofstream& fout);
            void load_model(std::ifstream& fin);
            // layers
            void append_batch_norm(void);
            void append_bconv(const int k, const int out_c, const int s=1);
            void append_gap(void);
            void loss_softmax_cross_entropy(void);
            void loss_weighted_softmax_cross_entropy(void);
            // getter setters
            void set_lr(float lr) {lr_ = lr;}
        
        // friend operators
        //friend std::ostream& operator<<(std::ostream& os, const Layer& l);

        private:
            // delete assigment
            Model& operator=(const Model& ref) = delete;
            // private functions
            void backward(Halide::Buffer<float> dldfx);
            void update(void);
            void forward_t(Halide::Buffer<float> in); 
            void register_last_layer(Layer* layer);
            void allocate_train_buffer(int batch_size);
            // private fields
            std::vector<bufdims> layer_out_dims_;
            std::vector< std::unique_ptr<Layer> > layer_ptrs_;
            std::unique_ptr<LossLayer> loss_layer_ptr_;
            bufdims in_dims_;
            bufdims out_dims_;
            bool training_;
            int batch_size_;
            float lr_;
            // buffer for inputs/ outputs of each layer
            std::unique_ptr<float[]> buf_i_;
            std::unique_ptr<float[]> buf_t_;
            std::vector<int> buf_offsets_;
    };
}

#endif