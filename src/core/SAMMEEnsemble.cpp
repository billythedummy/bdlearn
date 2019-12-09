#include "bdlearn/SAMMEEnsemble.hpp"

namespace bdlearn {

    // Destructor
    SAMMEEnsemble::~SAMMEEnsemble() {
        model_ptrs_.clear();
    }

    // public functions

    float SAMMEEnsemble::train_step() {
        dataset_->shuffle();
        Model* curr = model_ptrs_[current_m_i_].get();
        // get shuffled w to pass to model
        const int epoch_size = dataset_->get_epoch_size();
        const int epoch_steps = dataset_->get_steps();
        float w_shuffled [epoch_size];
        for (int i = 0; i < epoch_size; ++i) {
            //std::cout << dataset_->get_train_i()[i] << std::endl;
            w_shuffled[i] = w_[dataset_->get_train_i()[i]];
        }
        // iterate through one epoch
        int batch_offset = 0;
        for (int i = 0; i < epoch_steps; ++i) {
            batchdata batch = dataset_->get_next_batch();
            Halide::Buffer<float> x_batch (batch.x_ptr, in_dims_.w, in_dims_.h, in_dims_.c, batch.size);
            Halide::Buffer<float> y_batch (batch.y_ptr, classes_, batch.size);
            float loss = curr->train_step(x_batch, y_batch, static_cast<void*>(w_shuffled + batch_offset));
            std::cout << "Model " << current_m_i_ << " loss: " << loss << std::endl;
            batch_offset += batch.size;
            free_batch_data(batch);
        }
        // evaluate model and update w_ and alphas_
        float is_wrong_shuffled [epoch_size];
        batch_offset = 0;
        for (int i = 0; i < epoch_steps; ++i) {
            batchdata batch = dataset_->get_next_batch();
            Halide::Buffer<float> x_batch (batch.x_ptr, in_dims_.w, in_dims_.h, in_dims_.c, batch.size);
            Halide::Buffer<float> y_batch (batch.y_ptr, classes_, batch.size);
            curr->eval(x_batch, y_batch, static_cast<void*>(is_wrong_shuffled + batch_offset)); 
            batch_offset += batch.size;
            free_batch_data(batch);
        }

        // first rearrange is_wrong to unshuffled array
        // and calculate weighted error
        float is_wrong [epoch_size]; //wtv im not doing it in place
        float err = 0.0f;
        for (int i = 0; i < epoch_size; ++i) {
            int data_index = dataset_->get_train_i()[i];
            is_wrong[data_index] = is_wrong_shuffled[i];
            err += w_[data_index] * is_wrong[data_index];
        }

        // compute new alpha
        //assert(err > 0);
        //assert(err < 1);
        float alpha = logf((1-err) / err) + logf(classes_ - 1);
        alphas_[current_m_i_] = alpha;
        // compute new weights
        float total_w = 0.0f;
        for (int i = 0; i < epoch_size; ++i) {
            w_[i] *= expf(alpha * is_wrong[i]);
            total_w += w_[i];
        }
        // renormalize
        for (int i = 0; i < epoch_size; ++i) {
            w_[i] /= total_w;
        }
        // increment current_m_i_
        current_m_i_ = (current_m_i_ + 1) % model_ptrs_.size();
        return err;
    }

    void SAMMEEnsemble::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        // TO-DO
    }
    
    void SAMMEEnsemble::forward_batch(float* out, Halide::Buffer<float> in) {
        // TO-DO?
    }
            
    void SAMMEEnsemble::add_model(Model* model) {
        model_ptrs_.push_back(std::unique_ptr<Model>(model));
        alphas_.push_back(1.0f);
    }

    float SAMMEEnsemble::eval(DataSet* dataset) {
        const bufdims in_dims = dataset->get_x_dims();
        const int classes = dataset->get_classes();
        dataset->set_batch_size(batch_size_);
        float total_errors = 0.0f;
        for (int step = 0; step < dataset->get_steps(); ++step) {
            batchdata batch = dataset->get_next_batch();
            float batch_res [get_n_models() * batch.size * classes] = {0};
            float batch_max [batch.size * classes] = {0};
            int batch_amax [batch.size] = {0};

            Halide::Buffer<float> batch_max_view(batch_max, classes, batch_size_);
            Halide::Buffer<int> batch_amax_view(batch_amax, batch.size);
            Halide::RDom c_r(0, classes);
            Halide::Var c, b;
            for (unsigned int i = 0; i < model_ptrs_.size(); ++i) {
                float alpha = alphas_[i];
                float* batch_out = batch_res + i *batch.size*classes;
                Halide::Buffer<float> batch_in(batch.x_ptr, in_dims.w, in_dims.h, in_dims.c, batch.size);
                model_ptrs_[i]->forward_batch(batch_out, batch_in);
                // argmax
                Halide::Buffer<float> batch_out_view(batch_out, classes, batch.size);
                Halide::Func amax;
                amax(b) = Halide::argmax(batch_out_view(c_r, b))[0];
                amax.realize(batch_amax_view);
                // add alpha to it
                Halide::Func inc_alpha;
                inc_alpha(c, b) = Halide::select(c == batch_amax_view(b),
                                    batch_max_view(c, b) + alpha,
                                    batch_max_view(c, b));
                inc_alpha.parallel(b);
                inc_alpha.realize(batch_max_view);
            }
            // Check errors
            // argmax over batch_max
            Halide::Func wrong_f;
            Halide::Buffer<float> y_view(batch.y_ptr, classes, batch.size);
            Halide::Expr is_wrong = Halide::argmax(y_view(c_r, b))[0] != Halide::argmax(batch_max_view(c_r, b))[0];

            /*
            for (int i = 0; i < batch.size; ++i) {
                for (int j = 0; j < classes; ++j) {
                    std::cout << batch.y_ptr[i*classes + j] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            for (int i = 0; i < batch.size; ++i) {
                for (int j = 0; j < classes; ++j) {
                    std::cout << batch_max[i*classes + j] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            */

            wrong_f(b) = Halide::select(is_wrong, 1, 0);
            wrong_f.parallel(b);
            wrong_f.realize(batch_amax_view); // reusing amax_view here
            for (int i = 0; i < batch.size; ++i) {
                total_errors += batch_amax[i];
            }

            /*
            for (int i = 0; i < batch.size; ++i) {
                std::cout << batch_amax[i] << ", ";
            }
            std::cout << std::endl << std::endl;
            */

            free_batch_data(batch);
        }
        return total_errors / dataset->get_epoch_size();
    }

    void SAMMEEnsemble::set_batch_size(const int batch_size) {
        batch_size_ = batch_size;
        if (dataset_) {
            dataset_->set_batch_size(batch_size);
        }
    }

    void SAMMEEnsemble::set_dataset(DataSet* dataset) {
        dataset_ = dataset;
        // set dims
        in_dims_ = dataset->get_x_dims();
        classes_ = dataset->get_classes();
        // set w_
        const int n = dataset->get_epoch_size();
        float* w = new float[n];
        for (int i = 0; i < n; ++i) w[i] = 1.0f / n;
        w_.reset(w);
        // set batch size for dataset
        if (batch_size_) {
            dataset->set_batch_size(batch_size_);
        }
        // reset current_m_i
        current_m_i_ = 0;
    }
}