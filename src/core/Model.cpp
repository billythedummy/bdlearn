#include "bdlearn/Model.hpp"

namespace bdlearn {
    // destructor
    Model::~Model() {
        buf_i_.reset();
        buf_t_.reset();
        // layer_ptrs_
    }

    // public functions

    float Model::train_step(Halide::Buffer<float> X, Halide::Buffer<float> Y) {
        const int batch_size = X.dim(3).extent();
        if (batch_size != batch_size_) {
            batch_size_ = batch_size;
            allocate_train_buffer(batch_size_);
        }
        forward_t(X);
        bufdims last_layer_out_dims = layer_out_dims_.back();
        Halide::Buffer<float> last_layer_out(buf_t_.get() + buf_offsets_.back(),
                                            last_layer_out_dims.w,
                                            last_layer_out_dims.h,
                                            last_layer_out_dims.c,
                                            batch_size_);
        float loss = loss_layer_ptr_->forward_t(last_layer_out, Y);
        loss_layer_ptr_->backward(last_layer_out);
        backward(last_layer_out);
        update();
        return loss;
    }

    void Model::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {

    }

    void Model::append_batch_norm() {
        Layer* new_layer = new BatchNorm(out_dims_.c, training_);
        register_last_layer(new_layer);
    }

    void Model::append_bconv(const int k, const int s, const int out_c) {
        Layer* new_layer = new BConvLayer(k, s, out_dims_.c, out_c, training_);
        register_last_layer(new_layer);
    }

    void Model::loss_softmax_cross_entropy() {
        loss_layer_ptr_.reset(new SoftmaxCrossEntropy());
    }


    // private functions

    void Model::forward_t(Halide::Buffer<float> in) {
        for (unsigned int i = 0; i < layer_ptrs_.size(); ++i) {
            bufdims this_layer_out_dims = layer_out_dims_[i];
            Halide::Buffer<float> this_layer_out(buf_t_.get() + buf_offsets_[i],
                                                this_layer_out_dims.w,
                                                this_layer_out_dims.h,
                                                this_layer_out_dims.c,
                                                batch_size_);
            layer_ptrs_[i]->forward_t(this_layer_out, in);
            in = this_layer_out;
        }
    }

    void Model::backward(Halide::Buffer<float> dldfx) {
        for (unsigned int i = layer_ptrs_.size()-1; i > 0; --i) {
            bufdims this_layer_in_dims = layer_out_dims_[i-1];
            Halide::Buffer<float> this_layer_dldx(buf_t_.get() + buf_offsets_[i-1],
                                                this_layer_in_dims.w,
                                                this_layer_in_dims.h,
                                                this_layer_in_dims.c,
                                                batch_size_);
            layer_ptrs_[i]->backward(this_layer_dldx, dldfx);
            dldfx = this_layer_dldx;
        }
        // first layer
        float placeholder [in_dims_.w * in_dims_.h * in_dims_.c * batch_size_];
        Halide::Buffer<float> placeholder_view(placeholder, in_dims_.w, in_dims_.h, in_dims_.c, batch_size_);
        layer_ptrs_[0]->backward(placeholder_view, dldfx);
    }

    void Model::update() {
        for (auto const& layer_ptr: layer_ptrs_) {
            layer_ptr->update(lr_);
        }
    }

    void Model::register_last_layer(Layer* layer) {
        // update dims
        bufdims this_layer_in_dims = in_dims_;
        if (layer_out_dims_.size() > 0) {
            this_layer_in_dims = layer_out_dims_.back();
        }
        bufdims this_layer_out_dims = layer->calc_out_dim(this_layer_in_dims);
        layer_out_dims_.push_back(this_layer_out_dims);
        out_dims_ = this_layer_out_dims;
        // update layer_ptrs
        layer_ptrs_.push_back(std::unique_ptr<Layer>(layer));
    }

    void Model::allocate_train_buffer(int batch_size) {
        buf_offsets_.clear();
        int total_size = 0;
        bufdims in_dims = in_dims_;
        for (unsigned int i = 0; i < layer_ptrs_.size(); ++i) {
            buf_offsets_.push_back(total_size);
            bufdims layer_out_dims = layer_out_dims_[i];
            total_size += layer_out_dims.w * layer_out_dims.h * layer_out_dims.c * batch_size;
            in_dims = layer_out_dims;
        }
        buf_t_.reset(new float[total_size]);
    }
}