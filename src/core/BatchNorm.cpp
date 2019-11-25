#include "bdlearn/BatchNorm.hpp"

namespace bdlearn {
    // Constructors

    BatchNorm::BatchNorm(int channels) {
        channels_ = channels;
        // init gamma_ and var_ = 1
        float* gamma = new float[channels];
        Halide::Buffer<float> gamma_view(gamma, channels);
        float* var = new float[channels];
        Halide::Buffer<float> var_view(var, channels);
        Halide::Var i;
        Halide::Func one;
        one(i) = 1.0f;
        one.realize(gamma_view);
        gamma_.reset(gamma);
        one.realize(var_view);
        var_.reset(var);
        // init beta_ and mu_ = 0
        float* beta = new float[channels];
        Halide::Buffer<float> beta_view(beta, channels);
        float* mu = new float[channels];
        Halide::Buffer<float> mu_view(mu, channels);
        Halide::Var j;
        Halide::Func zero;
        zero(j) = 0.0f;
        zero.realize(beta_view);
        beta_.reset(beta);
        zero.realize(mu_view);
        mu_.reset(mu);
    }

    // Destructor

    BatchNorm::~BatchNorm() {
        gamma_.reset();
        beta_.reset();
        mu_.reset();
        var_.reset();
    }

    // public functions

    void BatchNorm::forward_t(Halide::Buffer<float>* out, Halide::Buffer<float> in) {
        Halide::Var c;
        int batch_size = in.dim(3).extent();
        int rows = in.dim(1).extent();
        int cols = in.dim(0).extent();
        int m = rows*cols*batch_size;
        Halide::RDom snb(0, rows, 0, cols, 0, batch_size); //space and batch
        // mean algo
        Halide::Func mean;
        Halide::Buffer<float> mu_view(mu_.get(), channels_);
        mean(c) = 0.0f;
        mean(c) += in(snb.x, snb.y, c, snb.z);
        mean(c) /= m;
        // mean schedule
        mean.realize(mu_view);
        // var algo
        Halide::Buffer<float> var_view(var_.get(), channels_);
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
        Halide::Expr var_plus_eps = var_view(c) + BDLEARN_EPS;
        x_hat(x, y, c, n) = ( in(x, y, c, n) - mu_view(c) ) * Halide::fast_inverse_sqrt(var_plus_eps);
        // output schedule
        x_hat.realize(*out);
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
}