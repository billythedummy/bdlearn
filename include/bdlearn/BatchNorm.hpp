#ifndef _BDLEARN_BATCHNORM_H_
#define _BDLEARN_BATCHNORM_H_

#include <cstddef>
#include <iostream>
#include "Halide.h"
#include "bdlearn/macros.hpp"
#include "bdlearn/Layer.hpp"

namespace bdlearn {
    class BatchNorm: public Layer {
        public:
        // Constructors
            // default - gamma, rvar, 1, beta, rmean 0
            BatchNorm(int channels, bool training=false);

        // Destructor
            virtual ~BatchNorm();

        // public functions
            void forward_t(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // training
            void forward_i(Halide::Buffer<float> out, Halide::Buffer<float> in) override; // inference
            void backward(Halide::Buffer<float> out, Halide::Buffer<float> ppg) override;
            bufdims calc_out_dim(bufdims in_dims) override;
            void update(float lr) override;
            void set_gamma(float* data);
            void set_beta(float* data);
            void set_r_mean(float* data);
            void set_r_var(float* data);
            float* get_r_mean(void);
            float* get_r_var(void);
            float* get_dgamma(void);
            float* get_dbeta(void);

        private:
            int channels_;
            std::unique_ptr<float[]> gamma_; // scale
            std::unique_ptr<float[]> beta_; // translation
            std::unique_ptr<float[]> r_mean_; // running mean
            std::unique_ptr<float[]> r_var_; // running variance
            // for storing gradients and other vars for backwards
            std::unique_ptr<float[]> mu_; // mean from prev input
            std::unique_ptr<float[]> var_; // variance from prev input
            std::unique_ptr<float[]> x_hat_; // normalized x from previous input
            std::unique_ptr<float[]> dbeta_; // dloss/dbeta
            std::unique_ptr<float[]> dgamma_; //dloss/dgamma
            Halide::Buffer<float> prev_in_; // previous input
    };
}

#endif