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
            curr->train_step(x_batch, y_batch, static_cast<void*>(w_shuffled + batch_offset)); //loss args
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

    void SAMMEEnsemble::save_ensemble(std::string path) {
        std::ofstream fout;
        fout.open(path, std::ios::out | std::ios::trunc);
        assert(!fout.fail());
        for (unsigned int i = 0; i < model_ptrs_.size(); ++i) {
            model_ptrs_[i]->save_model(fout);
        }
        fout.close();
    }

    void SAMMEEnsemble::load_ensemble(std::string path) {
        std::ifstream fin;
        fin.open(path, std::ios::in);
        assert(!fin.fail());
        for (unsigned int i = 0; i < model_ptrs_.size(); ++i) {
            model_ptrs_[i]->load_model(fin);
        }
        fin.close();
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