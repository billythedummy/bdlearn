#include "bdlearn/Model.hpp"

namespace bdlearn {
    // destructor
    Model::~Model() {}

    // public functions

    float Model::train_step(Halide::Buffer<float> X, Halide::Buffer<float> Y, void* loss_args) {
        const int batch_size = X.dim(3).extent();
        if (batch_size != batch_size_) {
            batch_size_ = batch_size;
            allocate_train_buffer(batch_size_);
        }
        forward_t(X);
        bufdims last_layer_out_dims = layer_out_dims_.back();
        Halide::Buffer<float> last_layer_out(buf_t_.back().get(),
                                            last_layer_out_dims.w,
                                            last_layer_out_dims.h,
                                            last_layer_out_dims.c,
                                            batch_size_);
        float loss = loss_layer_ptr_->forward_t(last_layer_out, Y, loss_args);
        loss_layer_ptr_->backward(last_layer_out);
        backward(last_layer_out);
        update();
        return loss;
    }

    void Model::forward_batch(float* out, Halide::Buffer<float> in) {
        const int batch_size = in.dim(3).extent();
        if (batch_size != batch_size_) {
            batch_size_ = batch_size;
            allocate_train_buffer(batch_size_);
        }
        forward_t(in);
        bufdims last_layer_out_dims = layer_out_dims_.back();
        const int n_floats = last_layer_out_dims.w * last_layer_out_dims.h * last_layer_out_dims.c * batch_size_;
        const float* src_ptr = buf_t_.back().get();
        memcpy(out, src_ptr, sizeof(float) * n_floats);
    }

    void Model::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {

    }

    float Model::eval(Halide::Buffer<float> X, Halide::Buffer<float> Y, void* is_wrong_out) {
        // classification error is the only evaluation func for now
        const int batch_size = X.dim(3).extent();
        if (batch_size != batch_size_) {
            batch_size_ = batch_size;
            allocate_train_buffer(batch_size_);
        }
        forward_t(X);
        bufdims last_layer_out_dims = layer_out_dims_.back();
        float* last_layer_ptr = buf_t_.back().get();
        Halide::Buffer<float> last_layer_view(last_layer_ptr, last_layer_out_dims.c, batch_size, "last_layer_view");
        float* out = static_cast<float*>(is_wrong_out);
        Halide::Buffer<float> out_view(out, batch_size, "out_view");
        Halide::Var b;
        Halide::RDom c_r(0, last_layer_out_dims.c);
        Halide::Func wrong_f;
        Halide::Expr is_wrong = Halide::argmax(Y(c_r, b))[0] != Halide::argmax(last_layer_view(c_r, b))[0];
        wrong_f(b) = Halide::select(is_wrong, 1.0f, 0.0f);
        // argmax realize
        wrong_f.realize(out_view);
        // return error rate
        float wrong = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            wrong += out[i];
        }
        return wrong / batch_size;
    }

    void Model::append_batch_norm() {
        Layer* new_layer = new BatchNorm(out_dims_.c, training_);
        register_last_layer(new_layer);
    }

    void Model::append_bconv(const int k, const int out_c, const int s) {
        Layer* new_layer = new BConvLayer(k, out_dims_.c, out_c, s, training_);
        register_last_layer(new_layer);
    }

    void Model::append_conv(const int k, const int out_c, const int s) {
        Layer* new_layer = new ConvLayer(k, out_dims_.c, out_c, s, training_);
        register_last_layer(new_layer);
    }

    void Model::append_gap() {
        Layer* new_layer = new GAP(training_);
        register_last_layer(new_layer);
    }

    void Model::append_max_pool(const int k) {
        Layer* new_layer = new MaxPool(k, k);
        register_last_layer(new_layer);
    }

    void Model::loss_softmax_cross_entropy() {
        loss_layer_ptr_.reset(new SoftmaxCrossEntropy());
    }

    void Model::loss_weighted_softmax_cross_entropy() {
        loss_layer_ptr_.reset(new WeightedSoftmaxCrossEntropy());
    }


    // private functions

    void Model::forward_t(Halide::Buffer<float> in) {
        for (unsigned int i = 0; i < layer_ptrs_.size(); ++i) {
            bufdims this_layer_out_dims = layer_out_dims_[i];
            Halide::Buffer<float> this_layer_out(buf_t_[i].get(),
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
            Halide::Buffer<float> this_layer_dldx(buf_t_[i-1].get(),
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
        buf_t_.clear();
        for (unsigned int i = 0; i < layer_ptrs_.size(); ++i) {
            bufdims layer_out_dims = layer_out_dims_[i];
            int layer_out_buf_size = layer_out_dims.w * layer_out_dims.h * layer_out_dims.c * batch_size;
            buf_t_.push_back(std::unique_ptr<float[]>(
                new float [layer_out_buf_size]
            ));
        }
    }
}