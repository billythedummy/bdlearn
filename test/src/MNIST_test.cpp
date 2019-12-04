#include "MNIST_test.hpp"

using namespace bdlearn;

int test_MNIST(void) {
    const int batch_size = 100; //<=100
    const int n_models = 1; // 1
    const bufdims in_dims = {.w=28, .h=28, .c=1};
    const int classes = 10;
    // make dataset
    DataSet ds;
    ds.load_darknet_classification("/home/dhy1996/uwimg/mnist.mini",
                                "/home/dhy1996/uwimg/mnist.labels");
    std::cout << "Loading dataset done" << std::endl;
    // make ensemble
    SAMMEEnsemble en(true);
    for (int i = 0; i < n_models; ++i) {
        Model* m = new Model(in_dims, true);
        m->append_batch_norm();
        m->append_bconv(5, 128); // 24
        m->append_batch_norm();
        m->append_bconv(3, 1024);
        m->append_gap();
        m->append_batch_norm();
        m->append_bconv(1, classes); 
        //m->loss_weighted_softmax_cross_entropy();
        m->loss_softmax_cross_entropy();
        /*
        m->append_batch_norm();
        m->append_bconv(5, 32);
        m->append_batch_norm();
        m->append_bconv(1, 16);
        m->append_batch_norm();
        m->append_bconv(5, 32);
        m->append_batch_norm();
        m->append_bconv(1, 16);
        m->append_batch_norm();
        m->append_bconv(3, 64);
        m->append_gap();
        m->append_batch_norm();
        m->append_bconv(1, classes);
        m->loss_softmax_cross_entropy();*/
        en.add_model(m);
    }
    en.set_batch_size(batch_size);
    en.set_dataset(&ds);
    en.set_lr(0.01f);
    // fuh reel
    for (int i=0; i < 100; ++i) {
        if (i % n_models == 0) {
            float err = en.eval(&ds);
            std::cout << "Current error rate: " << err << std::endl << std::endl;
            en.set_batch_size(batch_size);
        }
        float w_err = en.train_step();
        std::cout << "Model " << (i % n_models) << " Weighted error: " << w_err << std::endl;
        std::cout << "Curr Alphas: [";
        std::vector<float> alphas = en.get_alphas();
        for (auto& alpha: alphas) {
            std::cout << alpha << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }
    return 0;
}