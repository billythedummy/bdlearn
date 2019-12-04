#include "bdlearn/BatchNorm.hpp"

namespace bdlearn {

    const float BNORM_MOMENTUM = 0.1f;

    // Constructors

    BatchNorm::BatchNorm(int channels, bool training) {
        channels_ = channels;
        // init gamma_, var_ and r_var_ = 1
        float* gamma = new float[channels];
        Halide::Buffer<float> gamma_view(gamma, channels);
        float* r_var = new float[channels];
        Halide::Buffer<float> r_var_view(r_var, channels);
        Halide::Var i;
        Halide::Func one;
        one(i) = 1.0f;
        one.realize(gamma_view);
        gamma_.reset(gamma);
        one.realize(r_var_view);
        r_var_.reset(r_var);
        // init beta_, mu_ and r_mean_ = 0
        float* beta = new float[channels];
        Halide::Buffer<float> beta_view(beta, channels);
        float* r_mean = new float[channels];
        Halide::Buffer<float> r_mean_view(r_mean, channels);
        Halide::Var j;
        Halide::Func zero;
        zero(j) = 0.0f;
        zero.realize(beta_view);
        beta_.reset(beta);
        zero.realize(r_mean_view);
        r_mean_.reset(r_mean);
        if (training) {
            float* var = new float[channels];
            Halide::Buffer<float> var_view(var, channels);
            one.realize(var_view);
            var_.reset(var);
            float* mu = new float[channels];
            Halide::Buffer<float> mu_view(mu, channels);
            zero.realize(mu_view);
            mu_.reset(mu);
            dbeta_.reset(new float[channels]);
            dgamma_.reset(new float[channels]);
        }
    }

    // Destructor

    BatchNorm::~BatchNorm() {
        gamma_.reset();
        beta_.reset();
        mu_.reset();
        var_.reset();
        r_var_.reset();
        r_mean_.reset();
        dbeta_.reset();
        dgamma_.reset();
        x_hat_.reset();
    }

    // public functions

    void BatchNorm::forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        prev_in_ = in;
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

        // x_hat algo
        Halide::Var x, y, n;
        Halide::Func x_hat;
        // reset x_hat buffer in case of irregular batch size
        float* new_x_hat = new float[in.number_of_elements()];
        x_hat_.reset(new_x_hat);
        Halide::Buffer<float> x_hat_view(x_hat_.get(), cols, rows, channels_, batch_size);
        x_hat(x, y, c, n) = ( in(x, y, c, n) - mu_view(c) ) * Halide::fast_inverse_sqrt(var_view(c) + BDLEARN_EPS);
        // x_hat schedule
        
        Halide::Var xy;
        Halide::Expr vec_xy = rows * cols > 32 ? 32 : rows*cols;
        x_hat.fuse(x, y, xy);
        x_hat.vectorize(xy, vec_xy);
        x_hat.realize(x_hat_view);

        // output algo
        Halide::Func out_func;
        Halide::Buffer<float> gamma_view(gamma_.get(), channels_, "gamma_view");
        Halide::Buffer<float> beta_view(beta_.get(), channels_, "beta_view");
        out_func(x, y, c, n) = x_hat_view(x, y, c, n) * gamma_view(c) + beta_view(c);
        // output schedule
        
        out_func.fuse(x, y, xy);
        out_func.vectorize(xy, vec_xy);
        out_func.realize(out);

        // update running mean algo
        Halide::Expr vec_r = channels_ > 32 ? 32 : channels_;
        Halide::Buffer<float> r_mean_view(r_mean_.get(), channels_, "r_mean_view");
        Halide::Func update_r_mean;
        update_r_mean(c) = (1.0f - BNORM_MOMENTUM) * r_mean_view(c) + BNORM_MOMENTUM * mu_view(c);
        // update running mean schedule
        update_r_mean.vectorize(c, vec_r);
        update_r_mean.realize(r_mean_view);

        // update running var algo
        Halide::Buffer<float> r_var_view(r_var_.get(), channels_, "r_var_view");
        Halide::Func update_r_var;
        update_r_var(c) = (1.0f - BNORM_MOMENTUM) * r_var_view(c) + BNORM_MOMENTUM * var_view(c);
        // update running var schedule
        update_r_var.vectorize(c, vec_r);
        update_r_var.realize(r_var_view);
    }

    void BatchNorm::forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) {
        Halide::Var x, y, c;
        Halide::Func bnorm_i;
        Halide::Buffer<float> gamma_view(gamma_.get(), channels_);
        Halide::Buffer<float> beta_view(beta_.get(), channels_);
        Halide::Buffer<float> r_var_view(r_var_.get(), channels_);
        Halide::Buffer<float> r_mean_view(r_mean_.get(), channels_);
        Halide::Expr inv_std_dev = Halide::fast_inverse_sqrt(r_var_view(c) + BDLEARN_EPS);
        Halide::Expr t_correction = gamma_view(c) * r_mean_view(c) * inv_std_dev;
        bnorm_i(x, y, c) = in(x, y, c) * gamma_view(c) * inv_std_dev + beta_view(c) - t_correction;
        // schedule
        bnorm_i.realize(out);
    }

    void BatchNorm::backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) {
        // vars
        Halide::Var x, y, c, n;
        int batch_size = ppg.dim(3).extent();
        int rows = ppg.dim(1).extent();
        int cols = ppg.dim(0).extent();
        int m = rows*cols*batch_size;
        Halide::RDom snb(0, cols, 0, rows, 0, batch_size); //space and batch

        // dl/dbeta algo
        Halide::Buffer<float> dbeta_view(dbeta_.get(), channels_);
        Halide::Func dbeta_f;
        dbeta_f(c) = 0.0f;
        dbeta_f(c) += ppg(snb.x, snb.y, c, snb.z);
        // dl/dbeta schedule
        dbeta_f.realize(dbeta_view);

        // dl/dgamma algo
        Halide::Buffer<float> dgamma_view(dgamma_.get(), channels_);
        Halide::Buffer<float> x_hat_view(x_hat_.get(), cols, rows, channels_, batch_size);
        Halide::Func dgamma_f;
        dgamma_f(c) = 0.0f;
        dgamma_f(c) += ppg(snb.x, snb.y, c, snb.z) * x_hat_view(snb.x, snb.y, c, snb.z);
        // dl/dgamma schedule
        dgamma_f.realize(dgamma_view);

        // some common factors
        Halide::Buffer<float> var_view(var_.get(), channels_);
        Halide::Expr inv_std_dev = Halide::fast_inverse_sqrt(var_view(c) + BDLEARN_EPS);
        Halide::Buffer<float> mu_view(mu_.get(), channels_);

        // dl/dvar algo
        Halide::Buffer<float> gamma_view(gamma_.get(), channels_);
        Halide::Expr dvar_factor = (-gamma_view(c) / 2) * Halide::fast_pow(var_view(c) + BDLEARN_EPS, -1.5f);
        float dvar [channels_];
        Halide::Buffer<float> dvar_view(dvar, channels_);
        Halide::Func dvar_f;
        dvar_f(c) = 0.0f;
        dvar_f(c) += ppg(snb.x, snb.y, c, snb.z) * (prev_in_(snb.x, snb.y, c, snb.z) - mu_view(c));
        dvar_f(c) *= dvar_factor;
        // dl/dvar schedule
        dvar_f.realize(dvar_view);

        // dl/dmu algo
        float dmu [channels_];
        Halide::Buffer<float> dmu_view(dmu, channels_);
        Halide::Func dmu_lhs_f;
        dmu_lhs_f(c) = 0.0f;
        dmu_lhs_f(c) += ppg(snb.x, snb.y, c, snb.z) * -gamma_view(c) * inv_std_dev;
        dmu_lhs_f.realize(dmu_view);
        Halide::Func dmu_rhs_f;
        float dmu_rhs [channels_];
        Halide::Buffer<float> dmu_rhs_view(dmu_rhs, channels_);
        dmu_rhs_f(c) = 0.0f;
        dmu_rhs_f(c) += (dvar_view(c) / Halide::cast<float>(m)) * -2.0f * (prev_in_(snb.x, snb.y, c, snb.z) - mu_view(c));
        dmu_rhs_f.realize(dmu_rhs_view); // HOW TO REALIZE 2 REDUCTIONS OVER SAME DOMAIN SIMULTANEOUSLY? I DONT WANNA ALLOCATE 2 ARRAYS MANG
        Halide::Func dmu_f;
        dmu_f(c) = dmu_view(c) + dmu_rhs_view(c);
        dmu_f.realize(dmu_view);

        // dl/dx
        Halide::Func dx_f;
        Halide::Expr dx_hat = ppg(x, y, c, n) * gamma_view(c);
        Halide::Expr s_1 = dx_hat * inv_std_dev;
        Halide::Expr x_sub_mu = prev_in_(x, y, c, n) - mu_view(c);
        Halide::Expr s_2 = (dvar_view(c) * 2.0f * x_sub_mu) / Halide::cast<float>(m);
        Halide::Expr s_3 = dmu_view(c) / Halide::cast<float>(m);
        dx_f(x, y, c, n) = s_1 + s_2 + s_3;
        // dl/dx schedule
        dx_f.realize(out);
    }

    bufdims BatchNorm::calc_out_dim(bufdims in_dims) {
        return in_dims;
    }

    void BatchNorm::update(float lr) {
        // update gamma
        Halide::Var c;
        Halide::Func desc_gamma_f;
        Halide::Buffer<float> dgamma_view(dgamma_.get(), channels_);
        Halide::Buffer<float> gamma_view(gamma_.get(), channels_);
        desc_gamma_f(c) = gamma_view(c) - lr * dgamma_view(c);
        desc_gamma_f.realize(gamma_view);
        // update beta
        Halide::Func desc_beta_f;
        Halide::Buffer<float> dbeta_view(dbeta_.get(), channels_);
        Halide::Buffer<float> beta_view(beta_.get(), channels_);
        desc_beta_f(c) = beta_view(c) - lr * dbeta_view(c);
        desc_beta_f.realize(beta_view);
        /*
        std::cout << "dgamma: " << std::endl;
        for (int i = 0; i < channels_; ++i) {
            std::cout << dgamma_[i] << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "gamma: " << std::endl;
        for (int i = 0; i < channels_; ++i) {
            std::cout << gamma_[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "dbeta: " << std::endl;
        for (int i = 0; i < channels_; ++i) {
            std::cout << dbeta_[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "beta: " << std::endl;
        for (int i = 0; i < channels_; ++i) {
            std::cout << beta_[i] << ", ";
        }
        std::cout << std::endl;
        */
    }

    void BatchNorm::set_gamma(float* data) {
        for (int i = 0; i < channels_; ++i) {
            gamma_[i] = data[i];
        }
    }

    void BatchNorm::set_beta(float* data) {
        for (int i = 0; i < channels_; ++i) {
            beta_[i] = data[i];
        }
    }

    float* BatchNorm::get_r_mean() {
        return r_mean_.get();
    }
    float* BatchNorm::get_r_var() {
        return r_var_.get();
    }
    float* BatchNorm::get_dgamma() {
        return dgamma_.get();
    }
    float* BatchNorm::get_dbeta() {
        return dbeta_.get();
    }
}