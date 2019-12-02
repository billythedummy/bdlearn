#include "bdlearn/SoftmaxCrossEntropy.hpp"

namespace bdlearn {
    // Destructor
    SoftmaxCrossEntropy::~SoftmaxCrossEntropy() {
        q_.reset();
    }

    // public functions
    float SoftmaxCrossEntropy::forward_t(Halide::Buffer<float> in, Halide::Buffer<float> one_hot, void* args) {
        // one_hot Halide dims: classes, batch
        // in Halide dims: width=1, height=1, classes, batch
        one_hot_ = one_hot;
        const int classes = one_hot.dim(0).extent();
        const int batch = one_hot.dim(1).extent();
        assert(in.dim(2).extent() == classes);
        assert(in.dim(3).extent() == batch);
        assert(in.dim(0).extent() == 1);
        assert(in.dim(1).extent() == 1);
        // reshape to match in with one_hot
        float* in_begin = in.get()->begin(); // this is super hacky i know
        Halide::Buffer<float> in_view(in_begin, classes, batch);
        // calculate e^x
        float* q = new float[classes * batch];
        q_.reset(q);
        Halide::Buffer<float> q_view(q, classes, batch);
        Halide::Var c, b;
        Halide::Func exp_f;
        exp_f(c, b) = Halide::exp(in_view(c, b));
        exp_f.realize(q_view); // q_view is now e^x
        // calculate sum e^x
        float sum_q [batch];
        Halide::Buffer<float> sum_q_view(sum_q, batch);
        Halide::RDom cR(0, classes);
        Halide::Func sum_exp_f;
        sum_exp_f(b) = 0.0f;
        sum_exp_f(b) += q_view(cR, b);
        sum_exp_f.realize(sum_q_view); // sum_q_view is now sum e^x
        // calculate q
        Halide::Func q_f;
        q_f(c, b) = q_view(c, b) / sum_q_view(b);
        q_f.realize(q_view); // q_view is now e^x / sum e^x (softmax)
        // calculate log (sum e^x)
        Halide::Func log_f;
        log_f(b) = Halide::log(sum_q_view(b));
        log_f.realize(sum_q_view); // sum_q_view is now log (sum e^x)
        // calculate log q = x - log(sum e^x)
        float log_q [classes * batch];
        Halide::Buffer<float> log_q_view(log_q, classes, batch);
        Halide::Func log_q_f;
        log_q_f(c, b) = in_view(c, b) - sum_q_view(b);
        log_q_f.realize(log_q_view);
        // calculate total cross entropy = sum(one_hot * -log_q)
        float cross_entropy [batch];
        Halide::Buffer<float> cross_entropy_view(cross_entropy, batch);
        Halide::Func cross_entropy_f;
        cross_entropy_f(b) = 0.0f;
        cross_entropy_f(b) += one_hot(cR, b) * -log_q_view(cR, b);
        cross_entropy_f.realize(cross_entropy_view);
        // return mean of cross entropy as final loss
        float res = 0.0f;
        for (int i = 0; i < batch; ++i) {
            res += cross_entropy[i];
        }
        return res / batch;
    }

    void SoftmaxCrossEntropy::backward(Halide::Buffer<float> out_ppg) {
        // out_ppg Halide dims: width=1, height=1, classes, batch
        const int classes = one_hot_.dim(0).extent();
        const int batch = one_hot_.dim(1).extent();
        assert(out_ppg.dim(2).extent() == classes);
        assert(out_ppg.dim(3).extent() == batch);
        assert(out_ppg.dim(0).extent() == 1);
        assert(out_ppg.dim(1).extent() == 1);
        Halide::Var x, y, c, b;
        Halide::Buffer<float> q_view(q_.get(), classes, batch);
        Halide::Func mean_grad_f;
        mean_grad_f(x, y, c, b) = (q_view(c, b) - one_hot_(c, b)) / batch;
        mean_grad_f.realize(out_ppg);
    }

    float* SoftmaxCrossEntropy::get_q() {
        return q_.get();
    }

}