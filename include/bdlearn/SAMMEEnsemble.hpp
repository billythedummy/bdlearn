#ifndef _BDLEARN_SAMME_ENSEMBLE_H_
#define _BDLEARN_SAMME_ENSEMBLE_H_

#include "Halide.h"
#include "bdlearn/BufDims.hpp"
#include "bdlearn/Model.hpp"

namespace bdlearn {
    class SAMMEEnsemble {
        public:
        // constructor
            SAMMEEnsemble(const bool training)
            : training_(training), batch_size_(0),
            current_m_i_(0) {}

        // Destructor
            virtual ~SAMMEEnsemble();

        // public functions
            float train_step(void); // trains one model of the ensemble, returns error rate
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in); // inference
            void forward_batch(float* out, Halide::Buffer<float> in);
            void add_model(Model* model);
            // getter setters
            void set_lr(const float lr) {lr_ = lr;}
            void set_batch_size(const int batch_size) {batch_size_ = batch_size;}
            void set_dataset(float* X, float* Y, const int n, const bufdims in_dims, const bufdims out_dims);
            
        
        // friend operators
        //friend std::ostream& operator<<(std::ostream& os, const Layer& l);

        private:
            // delete assigment
            Model& operator=(const Model& ref) = delete;
            // private functions
            void shuffle_train_i(void);
            void batch_op_over_epoch(float (*op) (Halide::Buffer<float> x, Halide::Buffer<float> y, void* out),
                                    void* args = nullptr);
            // private fields
            std::vector< std::unique_ptr<Model> > model_ptrs_;
            std::vector<float> alphas_; // weights for each model
            bufdims in_dims_;
            bufdims out_dims_;
            // training params
            std::unique_ptr<float[]> train_X_;
            std::unique_ptr<float[]> train_Y_;
            std::unique_ptr<float[]> w_; // weights for each training example
            std::unique_ptr<int[]> train_i_; // order of iteration for training this epoch
            int epoch_size_;
            bool training_;
            int batch_size_;
            float lr_;
            int current_m_i_; // index of model currently being trained
    };
}

#endif