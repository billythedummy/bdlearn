#include "bdlearn/BatchNorm.hpp"

namespace bdlearn {

    const float BNORM_MOMENTUM = 0.1f;

    // Constructors

    BatchNorm::BatchNorm(int channels) {
        channels_ = channels;
        // init gamma_, var_ and r_var_ = 1
        float* gamma = new float[channels];
        Halide::Buffer<float> gamma_view(gamma, channels);
        float* var = new float[channels];
        Halide::Buffer<float> var_view(var, channels);
        float* r_var = new float[channels];
        Halide::Buffer<float> r_var_view(r_var, channels);
        Halide::Var i;
        Halide::Func one;
        one(i) = 1.0f;
        one.realize(gamma_view);
        gamma_.reset(gamma);
        one.realize(var_view);
        var_.reset(var);
        one.realize(r_var_view);
        r_var_.reset(r_var);
        // init beta_, mu_ and r_mean_ = 0
        float* beta = new float[channels];
        Halide::Buffer<float> beta_view(beta, channels);
        float* mu = new float[channels];
        Halide::Buffer<float> mu_view(mu, channels);
        float* r_mean = new float[channels];
        Halide::Buffer<float> r_mean_view(r_mean, channels);
        Halide::Var j;
        Halide::Func zero;
        zero(j) = 0.0f;
        zero.realize(beta_view);
        beta_.reset(beta);
        zero.realize(mu_view);
        mu_.reset(mu);
        zero.realize(r_mean_view);
        r_mean_.reset(r_mean);
    }

    // Destructor

    BatchNorm::~BatchNorm() {
        gamma_.reset();
        beta_.reset();
        mu_.reset();
        var_.reset();
        r_var_.reset();
        r_mean_.reset();
    }

    // public functions

    void BatchNorm::forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in) {
        Halide::Var c;
        int batch_size = in.dim(3).extent();
        int rows = in.dim(1).extent();
        int cols = in.dim(0).extent();
        int m = rows*cols*batch_size;
        Halide::RDom snb(0, cols, 0, rows, 0, batch_size); //space and batch
        // mean algo
        Halide::Func mean;
        Halide::Buffer<float> mu_view(mu_.get(), channels_, "mu_view");
        mean(c) = 0.0f;
        mean(c) += in(snb.x, snb.y, c, snb.z);
        mean(c) /= m;
        // mean schedule
        mean.realize(mu_view);
        // var algo
        Halide::Buffer<float> var_view(var_.get(), channels_, "var_view");
        Halide::Func var;
        var(c) = 0.0f;
        Halide::Expr diff = in(snb.x, snb.y, c, snb.z) - mu_view(c);
        var(c) += diff * diff;
        var(c) /= m;
        // var schedule
        var.realize(var_view);
        // output algo
        Halide::Var x, y, n;
        Halide::Func x_hat;
        x_hat(x, y, c, n) = ( in(x, y, c, n) - mu_view(c) ) * Halide::fast_inverse_sqrt(var_view(c) + BDLEARN_EPS);
        Halide::Func out_func;
        Halide::Buffer<float> gamma_view(gamma_.get(), channels_, "gamma_view");
        Halide::Buffer<float> beta_view(beta_.get(), channels_, "beta_view");
        out_func(x, y, c, n) = x_hat(x, y, c, n) * gamma_view(c) + beta_view(c);
        // output schedule
        out_func.realize(*out);
        // update running mean algo
        Halide::Buffer<float> r_mean_view(r_mean_.get(), channels_, "r_mean_view");
        Halide::Func update_r_mean;
        update_r_mean(c) = (1.0f - BNORM_MOMENTUM) * r_mean_view(c) + BNORM_MOMENTUM * mu_view(c);
        // update running mean schedule
        update_r_mean.realize(r_mean_view);
        // update running var algo
        Halide::Buffer<float> r_var_view(r_var_.get(), channels_, "r_var_view");
        Halide::Func update_r_var;
        update_r_var(c) = (1.0f - BNORM_MOMENTUM) * r_var_view(c) + BNORM_MOMENTUM * var_view(c);
        // update running var schedule
        update_r_var.realize(r_var_view);
    }

    void BatchNorm::forward_i(Halide::Buffer<float>* out, Halide::Buffer<float> in) {
        Halide::Var x, y, c;
        Halide::Func gam_x_pbeta;
        Halide::Buffer<float> gamma_view(gamma_.get(), channels_);
        Halide::Buffer<float> beta_view(beta_.get(), channels_);
        gam_x_pbeta(x, y, c) = in(x, y, c) * gamma_view(c) + beta_view(c);
        gam_x_pbeta.realize(*out);
    }

    void BatchNorm::backward(Halide::Buffer<float>* out, Halide::Buffer<float> ppg) {
        // TO-DO
        return;
    }

    void BatchNorm::set_gamma(float* data) {
        for (int i = 0; i < channels_; ++i) {
            gamma_.get()[i] = data[i];
        }
    }

    void BatchNorm::set_beta(float* data) {
        for (int i = 0; i < channels_; ++i) {
            beta_.get()[i] = data[i];
        }
    }

    float* BatchNorm::get_r_mean() {
        return r_mean_.get();
    }
    float* BatchNorm::get_r_var() {
        return r_var_.get();
    }
}