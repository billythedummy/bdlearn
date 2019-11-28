#include <iostream>
#include <random>
#include <chrono>
#include <math.h>
#include "BatchNorm_basic.hpp"

using namespace bdlearn;

int test_BatchNorm_forward_i() {
    int c = 7;
    BatchNorm dut(c);
    int m = 3;
    int n = 5;
    // random init test array
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / n));
    float test[c*m*n];
    for (int i = 0; i < c*m*n; ++i) test[i] = dist(generator);
    // test if output same with default
    Halide::Buffer<float> test_view(test, n, m, c);
    float out[c*m*n];
    Halide::Buffer<float> out_view(out, n, m, c);
    dut.forward_i(out_view, test_view);
    for (int i = 0; i < c*m*n; ++i) {
        if (fabsf(test[i] - out[i]) > 1E-3f) {
            std::cerr << "BatchNorm simple forward failed at " << i;
            std::cerr << " Expected: " << test[i] << ", got: " << out[i] << std::endl;
            return -1;
        }
    }
    return 0;
}

int test_BatchNorm_forward_backward_t() {
    int n = 3;
    int c = 2;
    int h = 7;
    int w = 5;
    float in [n*c*h*w] = {
        0.0747, -1.0128, -1.5366, -0.6547, -0.0991, -0.9915, -0.5302,  1.4154,
        0.5239,  0.9135,  0.6158, -0.2198, -0.1524,  0.3693,  2.1074,  0.5189,
        0.1102,  0.1779,  0.0671, -1.5374,  0.5950, -0.2125, -1.1538,  1.0116,
        -0.9214, -0.8291,  1.7052, -0.4510, -1.3789, -0.5126, -1.4952, -0.0753,
        -0.8261,  2.0631, -0.7567, -1.3826, -0.8150,  1.3471,  0.4798,  0.7179,
        1.1340,  0.5889, -0.2134, -0.6901, -0.1774,  1.0302, -0.4240,  0.5021,
        0.7628,  0.2325,  1.0409,  0.7585, -0.8585, -0.5446, -0.9727, -0.1291,
        2.6074, -2.2343,  0.6064, -0.3260, -0.2724, -0.0846, -0.5240, -1.2581,
        1.1995,  0.5046, -0.3891, -0.3375,  0.7471,  1.4137, -0.4667, -0.0663,
        -0.4460,  0.0204, -0.6131,  0.5144, -1.6866, -0.0449, -1.4782,  1.1493,
        -0.1753,  1.1916, -0.6927,  0.1872,  0.1564, -0.6036, -1.0734, -0.4008,
        0.0575, -2.0721, -0.7661, -0.4858,  0.6478, -0.7809, -1.8576, -1.3474,
        -0.2238,  0.0293,  1.2647,  0.4335, -0.0845,  0.9842, -1.1905, -0.1480,
        0.1682, -0.2174,  0.6433, -1.1864,  1.0854, -0.6009,  1.4719,  0.5848,
        0.8524, -1.0822, -1.3896, -0.0913,  0.5774,  1.8853, -0.6771, -0.6638,
        -1.6055,  0.5989, -0.0111,  0.6066, -0.4913, -0.5904, -1.4152, -1.6554,
        0.9123, -1.1299, -0.1538,  1.3556,  1.3582, -1.0184, -2.0924,  0.2108,
        1.5944,  0.3632, -2.0819, -0.6236,  1.7107, -0.0416, -0.5177, -0.9855,
        0.0831,  1.1793,  0.3095,  1.5436, -0.8943, -0.8034,  1.2099, -0.4950,
        -0.2701, -1.8806,  0.0626,  1.9288,  0.5609, -0.7070, -0.2094,  0.9855,
        0.0208,  0.4713, -0.7759, -0.5665,  0.1632,  0.9905,  1.2825,  0.6470,
        -0.3024, -0.9995,  2.4406,  0.9490, -0.5416, -0.4986,  1.5601, -1.1106,
        1.1531, -0.0778,  0.4435, -0.2745,  1.7231, -1.6106, -0.4124, -1.0898,
        -1.0184, -2.1986,  1.0782,  1.6674, -0.3935, -0.7654,  1.5076,  1.1080,
        0.3457,  0.7792,  0.8153,  0.5673, -0.4807, -1.5567, -0.4199, -0.6702,
        0.3996, -0.2271, -0.2982,  0.6324,  0.8332,  0.5566, -1.0110,  0.5762,
        0.4695, -0.2552
    };
    float expected_out [n*c*h*w] = {
        1.7945e-01, -3.3438e-01, -5.8187e-01, -1.6518e-01,  9.7332e-02,
        -3.2432e-01, -1.0636e-01,  8.1292e-01,  3.9169e-01,  5.7578e-01,
         4.3512e-01,  4.0302e-02,  7.2148e-02,  3.1865e-01,  1.1399e+00,
         3.8933e-01,  1.9622e-01,  2.2821e-01,  1.7586e-01, -5.8225e-01,
         4.2529e-01,  4.3752e-02, -4.0100e-01,  6.2213e-01, -2.9120e-01,
        -2.4759e-01,  9.4985e-01, -6.8938e-02, -5.0736e-01, -9.8043e-02,
        -5.6231e-01,  1.0858e-01, -2.4617e-01,  1.1190e+00, -2.1338e-01,
        -6.6446e-01, -2.5174e-01,  1.3204e+00,  6.8975e-01,  8.6288e-01,
         1.1654e+00,  7.6908e-01,  1.8570e-01, -1.6092e-01,  2.1188e-01,
         1.0900e+00,  3.2569e-02,  7.0597e-01,  8.9553e-01,  5.0993e-01,
         1.0977e+00,  8.9240e-01, -2.8337e-01, -5.5123e-02, -3.6641e-01,
         2.4700e-01,  2.2368e+00, -1.2838e+00,  7.8181e-01,  1.0383e-01,
         1.4280e-01,  2.7936e-01, -4.0144e-02, -5.7393e-01,  1.2131e+00,
         7.0779e-01,  5.7946e-02,  9.5466e-02,  8.8412e-01,  1.3688e+00,
        -7.6356e-02,  1.1283e-01, -6.6575e-02,  1.5379e-01, -1.4553e-01,
         3.8721e-01, -6.5275e-01,  1.2294e-01, -5.5428e-01,  6.8719e-01,
         6.1328e-02,  7.0718e-01, -1.8314e-01,  2.3261e-01,  2.1805e-01,
        -1.4104e-01, -3.6302e-01, -4.5219e-02,  1.7132e-01, -8.3489e-01,
        -2.1782e-01, -8.5380e-02,  4.5024e-01, -2.2481e-01, -7.3354e-01,
        -4.9248e-01,  3.8412e-02,  1.5800e-01,  7.4172e-01,  3.4898e-01,
         1.0423e-01,  6.0918e-01, -4.1835e-01,  7.4227e-02,  2.2363e-01,
         1.8279e-01,  8.0864e-01, -5.2180e-01,  1.1301e+00, -9.6061e-02,
         1.4111e+00,  7.6610e-01,  9.6068e-01, -4.4603e-01, -6.6955e-01,
         2.7449e-01,  7.6072e-01,  1.7117e+00, -1.5147e-01, -1.4180e-01,
        -8.2654e-01,  7.7635e-01,  3.3280e-01,  7.8195e-01, -1.6367e-02,
        -8.8426e-02, -6.8817e-01, -8.6282e-01,  1.0042e+00, -4.8071e-01,
         2.2904e-01,  1.3266e+00,  1.3285e+00, -3.9964e-01, -1.1806e+00,
         4.9415e-01,  1.5002e+00,  6.0497e-01, -1.1729e+00, -1.1257e-01,
         9.5245e-01,  1.2450e-01, -1.0045e-01, -3.2148e-01,  1.8342e-01,
         7.0137e-01,  2.9039e-01,  8.7349e-01, -2.7839e-01, -2.3544e-01,
         7.1582e-01, -8.9727e-02,  1.6536e-02, -7.4441e-01,  1.7373e-01,
         1.0555e+00,  4.0918e-01, -1.8990e-01,  4.5216e-02,  6.0980e-01,
         1.5398e-01,  3.6684e-01, -2.2245e-01, -1.2351e-01,  2.2127e-01,
         6.1216e-01,  7.5013e-01,  4.4986e-01,  1.2745e-03, -3.2810e-01,
         1.2973e+00,  5.9255e-01, -1.1175e-01, -9.1428e-02,  8.8129e-01,
        -4.6668e-01,  1.1793e+00,  2.8430e-01,  6.6336e-01,  1.4128e-01,
         1.5938e+00, -8.3025e-01,  4.1004e-02, -4.5156e-01, -3.9964e-01,
        -1.2578e+00,  1.1249e+00,  1.5533e+00,  5.4747e-02, -2.1567e-01,
         1.4371e+00,  1.1465e+00,  5.9224e-01,  9.0746e-01,  9.3371e-01,
         7.5338e-01, -8.6594e-03, -7.9106e-01,  3.5550e-02, -1.4645e-01,
         6.3144e-01,  1.7574e-01,  1.2404e-01,  8.0071e-01,  9.4672e-01,
         7.4560e-01, -3.9426e-01,  7.5985e-01,  6.8226e-01,  1.5531e-01
    };
    float gamma [c] = {0.45f, 0.73f};
    float beta [c] = {0.12f, 0.3419f};
    float expected_r_mean [c] = {-0.0051f,  0.0001f};
    float expected_r_var [c] = {0.9916, 1.0018};
    BatchNorm dut(c, true);
    dut.set_beta(beta);
    dut.set_gamma(gamma);
    Halide::Buffer<float> in_view(in, w, h, c, n, "batchnorm_t_in");
    float out [n*c*h*w];
    Halide::Buffer<float> out_view(out, w, h, c, n, "batchnorm_t_out");
    dut.forward_t(out_view, in_view);
    // verify data
    for (int i=0; i < n*c*h*w; ++i) {
        if (fabsf(out[i] - expected_out[i]) > 1E-3 ) {
            std::cerr << "test_BatchNorm_forward_t failed at " << i;
            std::cerr << ". Expected: " << expected_out[i] << ", got: " << out[i] << std::endl;
            return -1;
        }
    }
    // verify running mean
    float* r_mean = dut.get_r_mean();
    for (int i=0; i < c; ++i) {
        //std::cout << r_mean[i] << ", " << expected_r_mean[i] << std::endl;
        if (fabsf(r_mean[i] - expected_r_mean[i]) > 5E-3) {
            std::cerr << "test_BatchNorm_forward_t check r_mean failed at " << i;
            std::cerr << ". Expected: " << expected_r_mean[i] << ", got: " << r_mean[i] << std::endl;
            return -1;
        }
    }
    // verify running var
    float* r_var = dut.get_r_var();
    for (int i=0; i < c; ++i) {
        //std::cout << r_var[i] << ", " << expected_r_var[i] << std::endl;
        if (fabsf(r_var[i] - expected_r_var[i]) > 5E-3f) {
            std::cerr << "test_BatchNorm_forward_t check r_var failed at " << i;
            std::cerr << ". Expected: " << expected_r_var[i] << ", got: " << r_var[i] << std::endl;
            return -1;
        }
    }
    float dldx [n*c*h*w];
    Halide::Buffer<float> dldx_view(dldx, w, h, c, n);
    float ppg [n*c*h*w];
    for (int i = 0; i < n*c*h*w; ++i) {
        ppg[i] = 1.0f;
    }
    Halide::Buffer<float> ppg_view(ppg, w, h, c, n);
    dut.backward(dldx_view, ppg_view);
    for (int i = 0; i < n*c*h*w; ++i) {
        //std::cout << dldx[i] << ", ";
    }
    std::cout << std::endl;
    return 0;
}