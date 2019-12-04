#ifndef _BDLEARN_SAMME_ENSEMBLE_H_
#define _BDLEARN_SAMME_ENSEMBLE_H_

#include "Halide.h"
#include "bdlearn/BufDims.hpp"
#include "bdlearn/Model.hpp"
#include "bdlearn/DataSet.hpp"

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
            float eval(DataSet* dataset);
            // getter setters
            void set_lr(const float lr) {for (auto& model_ptr: model_ptrs_) model_ptr->set_lr(lr);}
            void set_batch_size(const int batch_size);
            void set_dataset(DataSet* dataset);
            int get_n_models(void) {return model_ptrs_.size();}
            // for debugging
            float* get_w(void) {return w_.get();}
            std::vector<float> get_alphas(void) {return alphas_;} // copies?
        
        // friend operators
        //friend std::ostream& operator<<(std::ostream& os, const Layer& l);

        private:
            // delete assigment
            Model& operator=(const Model& ref) = delete;
            // private fields
            std::vector< std::unique_ptr<Model> > model_ptrs_;
            std::vector<float> alphas_; // weights for each model
            bufdims in_dims_;
            int classes_;
            // training params
            std::unique_ptr<float[]> w_; // weights for each training example
            DataSet* dataset_;
            bool training_;
            int batch_size_;
            float lr_;
            int current_m_i_; // index of model currently being trained
    };
}

#endif