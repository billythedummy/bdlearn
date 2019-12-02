#include "bdlearn/SAMMEEnsemble.hpp"

namespace bdlearn {

    // Destructor
    SAMMEEnsemble::~SAMMEEnsemble() {
        model_ptrs_.clear();
    }

    // public functions

    float SAMMEEnsemble::train_step() {
        shuffle_train_i();
        Model* curr = model_ptrs_[current_m_i_].get();
        // get shuffled w to pass to model
        float w_shuffled [epoch_size_];
        for (int i = 0; i < epoch_size_; ++i) {
            w_shuffled[i] = w_[train_i_[i]];
        }

        // buffer and dim vars
        const int steps_m1 = epoch_size_ / batch_size_;
        const int x_offset = in_dims_.w * in_dims_.h * in_dims_.c;
        const int y_offset = out_dims_.w * out_dims_.h * out_dims_.c;
        const int rem = epoch_size_ % batch_size_;
        float x_buffer [batch_size_ * x_offset];
        float y_buffer [batch_size_ * y_offset];

        // iterate through one epoch
        for (int i = 0; i < steps_m1; ++i) {
            int batch_offset = i * batch_size_;
            for (int j = 0; j < batch_size_; ++j) {
                int curr_index = batch_offset + j;
                int data_index = train_i_[curr_index];
                memcpy(x_buffer + j*x_offset, train_X_+data_index*x_offset,
                        sizeof(float) * x_offset);
                memcpy(y_buffer + j*y_offset, train_Y_+data_index*y_offset,
                        sizeof(float) * y_offset);
            }
            Halide::Buffer<float> x_batch (x_buffer, in_dims_.w, in_dims_.h, in_dims_.c, batch_size_);
            Halide::Buffer<float> y_batch (y_buffer, out_dims_.c, batch_size_);
            curr->train_step(x_batch, y_batch, static_cast<void*>(w_shuffled + batch_offset)); //loss args
        }
        // in case of remainder for batch size
        if (rem) {
            for (int j = (epoch_size_ - rem); j < epoch_size_; ++j) {
                int data_index = train_i_[j];
                int j_zero_relative = j - epoch_size_ + rem;
                memcpy(x_buffer + j_zero_relative*x_offset, train_X_+data_index*x_offset,
                        sizeof(float) * x_offset);
                memcpy(y_buffer + j_zero_relative*y_offset, train_Y_+data_index*y_offset,
                        sizeof(float) * y_offset);
            }
            Halide::Buffer<float> x_batch (x_buffer, in_dims_.w, in_dims_.h, in_dims_.c, rem);
            Halide::Buffer<float> y_batch (y_buffer, out_dims_.c, rem);
            curr->train_step(x_batch, y_batch, static_cast<void*>(w_shuffled + epoch_size_ - rem)); //loss args  
        } 
        // evaluate model and update w_ and alphas_
        float is_wrong_shuffled [epoch_size_];
        for (int i = 0; i < steps_m1; ++i) {
            int batch_offset = i * batch_size_;
            for (int j = 0; j < batch_size_; ++j) {
                int curr_index = batch_offset + j;
                int data_index = train_i_[curr_index];
                memcpy(x_buffer + j*x_offset, train_X_+data_index*x_offset,
                        sizeof(float) * x_offset);
                memcpy(y_buffer + j*y_offset, train_Y_+data_index*y_offset,
                        sizeof(float) * y_offset);
            }
            Halide::Buffer<float> x_batch (x_buffer, in_dims_.w, in_dims_.h, in_dims_.c, batch_size_);
            Halide::Buffer<float> y_batch (y_buffer, out_dims_.c, batch_size_);
            curr->eval(x_batch, y_batch, static_cast<void*>(is_wrong_shuffled + batch_offset)); 
        }
        // in case of remainder for batch size
        if (rem) {
            for (int j = (epoch_size_ - rem); j < epoch_size_; ++j) {
                int data_index = train_i_[j];
                int j_zero_relative = j - epoch_size_ + rem;
                memcpy(x_buffer + j_zero_relative*x_offset, train_X_+data_index*x_offset,
                        sizeof(float) * x_offset);
                memcpy(y_buffer + j_zero_relative*y_offset, train_Y_+data_index*y_offset,
                        sizeof(float) * y_offset);
            }
            Halide::Buffer<float> x_batch (x_buffer, in_dims_.w, in_dims_.h, in_dims_.c, rem);
            Halide::Buffer<float> y_batch (y_buffer, out_dims_.c, rem);
            curr->eval(x_batch, y_batch, static_cast<void*>(is_wrong_shuffled + epoch_size_ - rem));
        }

        // first rearrange is_wrong to unshuffled array
        // and calculate weighted error
        float is_wrong [epoch_size_]; //wtv im not doing it in place
        float err = 0.0f;
        for (int i = 0; i < epoch_size_; ++i) {
            int data_index = train_i_[i];
            is_wrong[data_index] = is_wrong_shuffled[i];
            err += w_[data_index] * is_wrong[data_index];
        }

        // compute new alpha
        float alpha = logf((1-err) / err) + logf(out_dims_.c - 1);
        alphas_[current_m_i_] = alpha;
        // compute new weights
        float total_w = 0.0f;
        for (int i = 0; i < epoch_size_; ++i) {
            w_[i] *= expf(alpha * is_wrong[i]);
            total_w += w_[i];
        }
        // renormalize
        for (int i = 0; i < epoch_size_; ++i) {
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

    void SAMMEEnsemble::set_dataset(float* X, float* Y,
                                    const int n, const bufdims in_dims, const bufdims out_dims) {
        // set dims
        in_dims_ = in_dims;
        out_dims_ = out_dims;
        epoch_size_ = n;
        // set data pointers
        train_X_ = X;
        train_Y_ = Y;
        // set train_i_
        int* train_i = new int[n];
        for (int i = 0; i < n; ++i) train_i[i] = i;
        train_i_.reset(train_i);
        // set w_
        float* w = new float[n];
        for (int i = 0; i < n; ++i) w[i] = 1.0f / n;
        w_.reset(w);
    }

    // private functions 

    void SAMMEEnsemble::shuffle_train_i() {
        // Fisher-yates shuffle train_i_
        for (int i = 0; i < epoch_size_-1; ++i) {
            int swap_index = rand() % (epoch_size_-i) + i;
            int swap = train_i_[swap_index];
            train_i_[swap_index] = train_i_[i];
            train_i_[i] = swap;
        }
    }

}