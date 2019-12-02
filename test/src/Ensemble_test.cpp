#include "Ensemble_test.hpp"

using namespace bdlearn;

int test_Ensemble() {
    // dims 
    const int in_w = 5;
    const int in_h = 5;
    const int in_c = 3;
    const bufdims in_dims = {in_w, in_h, in_c};
    const int classes = 3;
    const bufdims out_dims = {1, 1, classes};
    const int batch_size = 64;
    const int epoch_size = 150;
    const int n_models = 5;
    const int X_size = epoch_size * in_w * in_h * in_c;
    const int Y_size = epoch_size * classes;
    const float LR = 1E-7f;
    // Create the ensemble
    SAMMEEnsemble dut(true);
    for (int i = 0; i < n_models; ++i) {
        Model* m = new Model(in_dims, true);
        m->append_batch_norm();
        m->append_bconv(3, 7);
        m->append_batch_norm();
        m->append_bconv(1, 16);
        m->append_batch_norm();
        m->append_bconv(3, classes);
        m->loss_weighted_softmax_cross_entropy();
        dut.add_model(m);
    }
    // generate fake data
    float X [X_size];
    for (int i = 0; i < X_size; ++i) {
        X[i] = i - X_size / 2 + 0.1f;
    }
    float Y [Y_size];
    int ind = 0;
    for (int i = 0; i < epoch_size; ++i) {
        for (int j = 0; j < classes; ++j) {
            Y[i*classes + j] = 0;
        }
        Y[i*classes + ind] = 1;
        ind = (ind + 1) % classes;
    }
    dut.set_dataset(X, Y, epoch_size, in_dims, out_dims);
    dut.set_batch_size(batch_size);
    dut.set_lr(LR);
    // train all my homies
    for (int i=0; i < 5*n_models; ++i) {
        float w_err = dut.train_step();
        std::cout << "Current weighted error: " << w_err << std::endl;
        /*
        std::cout << "Curr W: [";
        float* w = dut.get_w();
        for (int j = 0; j < epoch_size; ++j) {
            std::cout << w[j] << ", ";
        }
        std::cout << "]" << std::endl;*/
        std::cout << "Curr Alphas: [";
        std::vector<float> alphas = dut.get_alphas();
        for (auto& alpha: alphas) {
            std::cout << alpha << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }
    return 0;
}