#include "Softmax_test.hpp"

using namespace bdlearn;

int test_softmax() {
    const int batch = 3;
    const int classes = 10;
    float one_hot [batch*classes] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    }; // 8, 7, 0
    float fx [batch*classes] = {
        -1.34340341, -0.45974003, -1.13929274, -0.70190898, -0.44352927,
        0.75805997,  0.57303502, -0.48544892, -0.84611373,  2.63468655,
       -0.89735092, -0.68660326, -0.17047781,  0.71524846, -1.53322612,
        1.51689624, -1.01533733, -0.34391119, -0.48589473, -0.11330478,
        0.39868253, -1.51340956,  0.66346402,  0.68727878,  2.15907718,
        1.19633974,  0.46431238, -1.61770077, -0.02563707,  0.70544018
    };
    float expected_loss = 3.12265;
    float expected_q [batch*classes] = {0.012285447, 0.029727679, 0.015067288, 0.023333956, 0.030213514, 0.10047196, 0.08350059, 0.02897316, 0.020200454, 0.656226, 0.036553558, 0.045129046, 0.07561476, 0.1833465, 0.01935408, 0.40871805, 0.032485437, 0.063574865, 0.055159815, 0.080063865, 0.06651268, 0.009828638, 0.08667574, 0.08876467, 0.38675335, 0.14768028, 0.07102431, 0.008855239, 0.043513566, 0.090391494};
    float expected_grad [batch*classes] = {0.0040951488, 0.009909227, 0.0050224294, 0.007777985, 0.010071171, 0.033490654, 0.027833529, 0.00965772, -0.32659987, 0.218742, 0.012184519, 0.015043016, 0.02520492, 0.0611155, 0.0064513604, 0.13623935, 0.010828479, -0.31214172, 0.018386604, 0.026687955, -0.31116244, 0.0032762128, 0.028891914, 0.029588223, 0.12891778, 0.04922676, 0.023674771, 0.002951746, 0.014504522, 0.030130498};
    
    SoftmaxCrossEntropy dut;
    Halide::Buffer<float> fx_view(fx, 1, 1, classes, batch);
    Halide::Buffer<float> one_hot_view(one_hot, classes, batch);
    float loss = dut.forward_t(fx_view, one_hot_view);
    if (fabsf(loss - expected_loss) > 1E-3f) {
        std::cout << "softmax test failed: ";
        std::cout << "Expected loss: " << expected_loss << ". Got: " << loss << std::endl;
        return -1;
    }
    float* q = dut.get_q();
    for (int i = 0; i < batch*classes; ++i) {
        if (fabsf(q[i] - expected_q[i]) > 1E-3f) {
            std::cout << "softmax test failed q at index " << i;
            std::cout << ". Expected: " << expected_q[i] << ". Got: " << q[i] << std::endl;
            return -1;
        }
    }
    float grad [batch*classes];
    Halide::Buffer<float> grad_view(grad, 1, 1, classes, batch);
    dut.backward(grad_view);
    for (int i = 0; i < batch*classes; ++i) {
        if (fabsf(grad[i] - expected_grad[i]) > 1E-3f) {
            std::cout << "softmax test failed grad at index " << i;
            std::cout << ". Expected: " << expected_grad[i] << ". Got: " << grad[i] << std::endl;
            return -1;
        }
    }
    return 0;
}