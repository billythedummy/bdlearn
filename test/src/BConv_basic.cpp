#include <iostream>
#include <fstream>
#include "BConv_basic.hpp"

using namespace bdlearn;

int test_BConv_rand_constructor() {
    int k = 13;
    int in_c = 3;
    int out_c = 5;
    BConvLayer dut (k, in_c, out_c, 1);
    for (int out_c_i = 0; out_c_i < out_c; ++out_c_i) {
        for (int in_c_i = 0; in_c_i < in_c; ++in_c_i) {
            for (int y = 0; y < k; ++y) {
                for (int x = 0; x < k; ++x) {
                    if (dut.get_w(x, y, in_c_i, out_c_i)
                        != (dut.get_train_w(x, y, in_c_i, out_c_i) >= 0)) {
                        std::cerr << "test_BConv_rand_constructor failed at " << out_c_i << ", " << in_c_i << ", " << y << ", " << x;
                        std::cerr << ". Expected: " << (dut.get_train_w(x, y, in_c_i, out_c_i) >= 0) << ", got: " << dut.get_w(x, y, in_c_i, out_c_i) << std::endl;
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

int test_forward_i() {
    const int s = 1;
    const int k = 3;
    const int width = 4;
    const int height = 5;
    const int out_height = (height - k) / s + 1;
    const int out_width = (width - k) / s + 1;
    const int in_c = 3;
    const int out_c = 4;
    float out [out_c * out_height * out_width];
    Halide::Buffer<float> out_view(out, out_width, out_height, out_c);
    float in [in_c * width * height] = {
        -1.00865331, -0.29219059,  2.67323418,  1.8020211 ,  0.06334328,
       -0.93614211, -0.0566484 ,  0.10741072, -0.35934454, -2.38577151,
       -1.12968527, -1.73120435, -0.99528351,  0.14717974,  1.10746035,
        0.12275506, -1.23225989, -0.91246136,  0.57759139,  0.14330837,
        0.19972974,  0.23643302,  1.78377793, -1.63716193, -0.71179966,
        0.94430747,  1.31911213, -0.84233116,  0.11643658, -1.69246736,
       -1.64737212,  1.08346985, -0.06631668, -0.62604917,  0.52611927,
       -0.31612989, -1.61702444,  0.99476058, -0.55800325,  2.00426533,
        3.442167  , -1.02430882,  0.09381595, -0.7679083 ,  2.014368  ,
       -0.07990519, -0.20638353,  0.87695587, -2.87695277,  0.63686711,
       -0.37224111,  0.30792107,  0.47311558,  0.53818892, -1.02746744,
       -0.42890086, -0.3766753 ,  0.13679642, -0.32584462,  0.19585104
    };
    float W [k*k*in_c*out_c] = {
        -0.7299722 , -0.78967849,  0.84872669, -0.49856974, -1.09713458,
        1.47446225, -0.48514377, -0.12391957, -0.32268059,  0.34577184,
       -1.07259811,  1.85186512, -1.43826704,  0.37966712,  0.77195291,
        0.08796895, -0.51668848,  0.61286801,  0.59773526, -1.09789223,
       -0.57357792, -0.49615405, -0.61383422, -0.87604905, -0.12136966,
        0.84372212,  2.14358485,  0.30483449,  1.00991064,  0.38306488,
        0.59696656,  0.87319798, -0.06306927, -1.4817166 ,  2.28488271,
       -0.9710247 ,  0.32753392,  1.54709241, -0.1421556 ,  0.52972941,
       -0.85173797,  0.2514344 ,  1.38198671,  0.56488595, -0.27808375,
        0.30079492, -1.5604293 , -0.76920338,  0.94884777, -0.34898017,
       -0.27261119,  0.66253144, -0.44718849, -1.04649192, -0.40443405,
        0.75018633,  0.76379933, -0.18070994,  1.69631008,  0.95052768,
       -0.49252714,  0.09670723,  1.18107807, -0.60567241, -1.61544695,
       -0.70376182, -0.53966195, -0.38910442, -0.14767869, -0.32604882,
        0.03504815,  0.53011946,  0.99830563,  0.2980226 ,  0.18446012,
        0.35257727, -0.39835519,  1.29643823, -0.18279839, -0.80132703,
       -0.13609081,  0.35219595, -0.58026604,  1.25114061,  0.59609455,
       -0.49310708,  0.03367772, -0.29258327, -0.46005861, -2.14134511,
        0.38816264, -2.92246895,  1.87465183, -0.43234896,  0.94031102,
       -0.13359161, -0.59917379, -1.22028055,  1.16533403, -0.71487912,
       -0.13869648, -0.72062408, -0.01038284, -1.82785722,  0.77738807,
        0.03013952, -0.03744685, -0.56171895
    };
    float expected_out[out_c * out_height * out_width] = {13.0, 5.0, -1.0, -7.0, 1.0, 5.0, 5.0, -3.0, -1.0, 5.0, -3.0, -3.0, -7.0, 1.0, -5.0, 5.0, 5.0, 5.0, 3.0, 11.0, 1.0, -1.0, -5.0, 3.0};
    Halide::Buffer<float> in_view(in, width, height, in_c);
    BConvLayer dut (k, in_c, out_c, 1, true);
    dut.load_weights(W);
    dut.forward_i(out_view, in_view);
    for (int i = 0; i < out_c * out_height * out_width; ++i) {
        if (fabsf(out[i] - expected_out[i]) > 1E-3f) {
            std::cerr << "conv_forward_backward failed at forward_t " << i;
            std::cerr << ". Expected: " << expected_out[i] << " got: " << out[i] << std::endl;
            return -1;
        }
    }
    return 0;
}

int test_BConv_forward_backward() {
    const int s = 1;
    const int k = 3;
    const int width = 4;
    const int height = 5;
    const int out_height = (height - k) / s + 1;
    const int out_width = (width - k) / s + 1;
    const int in_c = 3;
    const int out_c = 4;
    const int batch = 2;
    float out [batch * out_c * out_height * out_width];
    Halide::Buffer<float> out_view(out, out_width, out_height, out_c, batch);
    float in [batch * in_c * width * height] = {
        -1.00865331, -0.29219059,  2.67323418,  1.8020211 ,  0.06334328,
       -0.93614211, -0.0566484 ,  0.10741072, -0.35934454, -2.38577151,
       -1.12968527, -1.73120435, -0.99528351,  0.14717974,  1.10746035,
        0.12275506, -1.23225989, -0.91246136,  0.57759139,  0.14330837,
        0.19972974,  0.23643302,  1.78377793, -1.63716193, -0.71179966,
        0.94430747,  1.31911213, -0.84233116,  0.11643658, -1.69246736,
       -1.64737212,  1.08346985, -0.06631668, -0.62604917,  0.52611927,
       -0.31612989, -1.61702444,  0.99476058, -0.55800325,  2.00426533,
        3.442167  , -1.02430882,  0.09381595, -0.7679083 ,  2.014368  ,
       -0.07990519, -0.20638353,  0.87695587, -2.87695277,  0.63686711,
       -0.37224111,  0.30792107,  0.47311558,  0.53818892, -1.02746744,
       -0.42890086, -0.3766753 ,  0.13679642, -0.32584462,  0.19585104,
        0.70778735, -2.49214661, -0.97716056, -0.81333065, -2.03955354,
        0.63255747, -0.58223596,  0.74004346,  0.9673292 ,  1.05641445,
        0.32640769,  0.16965868,  0.06097121, -0.37424648,  0.83535751,
       -0.22970285, -1.29852304,  1.60117816, -0.4266019 ,  0.03812115,
       -1.62695264, -0.45039727, -0.89233058, -0.34620122, -0.3616281 ,
        1.45834436,  0.92573991, -1.25538252, -0.58039008, -0.04393592,
        0.11642086,  0.0212527 ,  0.32238868, -0.70639151, -0.53507394,
       -0.08731686, -0.2609255 , -0.42632088,  1.02653768, -0.54569219,
        0.14520752, -0.79650049, -1.3959103 , -1.21162863,  0.37943659,
       -1.84386534, -2.54894931,  0.00668073, -0.36955124,  1.77577916,
        0.46833654, -0.22232106, -1.01085293, -0.0624432 ,  1.02065703,
       -1.22657194, -0.26174993,  0.0906495 ,  2.52473708,  1.67438017
    };
    float W [k*k*in_c*out_c] = {
        -0.7299722 , -0.78967849,  0.84872669, -0.49856974, -1.09713458,
        1.47446225, -0.48514377, -0.12391957, -0.32268059,  0.34577184,
       -1.07259811,  1.85186512, -1.43826704,  0.37966712,  0.77195291,
        0.08796895, -0.51668848,  0.61286801,  0.59773526, -1.09789223,
       -0.57357792, -0.49615405, -0.61383422, -0.87604905, -0.12136966,
        0.84372212,  2.14358485,  0.30483449,  1.00991064,  0.38306488,
        0.59696656,  0.87319798, -0.06306927, -1.4817166 ,  2.28488271,
       -0.9710247 ,  0.32753392,  1.54709241, -0.1421556 ,  0.52972941,
       -0.85173797,  0.2514344 ,  1.38198671,  0.56488595, -0.27808375,
        0.30079492, -1.5604293 , -0.76920338,  0.94884777, -0.34898017,
       -0.27261119,  0.66253144, -0.44718849, -1.04649192, -0.40443405,
        0.75018633,  0.76379933, -0.18070994,  1.69631008,  0.95052768,
       -0.49252714,  0.09670723,  1.18107807, -0.60567241, -1.61544695,
       -0.70376182, -0.53966195, -0.38910442, -0.14767869, -0.32604882,
        0.03504815,  0.53011946,  0.99830563,  0.2980226 ,  0.18446012,
        0.35257727, -0.39835519,  1.29643823, -0.18279839, -0.80132703,
       -0.13609081,  0.35219595, -0.58026604,  1.25114061,  0.59609455,
       -0.49310708,  0.03367772, -0.29258327, -0.46005861, -2.14134511,
        0.38816264, -2.92246895,  1.87465183, -0.43234896,  0.94031102,
       -0.13359161, -0.59917379, -1.22028055,  1.16533403, -0.71487912,
       -0.13869648, -0.72062408, -0.01038284, -1.82785722,  0.77738807,
        0.03013952, -0.03744685, -0.56171895
    };
    float expected_out[batch * out_c * out_height * out_width] = {13.0, 5.0, -1.0, -7.0, 1.0, 5.0, 5.0, -3.0, -1.0, 5.0, -3.0, -3.0, -7.0, 1.0, -5.0, 5.0, 5.0, 5.0, 3.0, 11.0, 1.0, -1.0, -5.0, 3.0, 5.0, -3.0, 1.0, 1.0, 1.0, -5.0, -3.0, -7.0, -3.0, 5.0, -7.0, -1.0, 1.0, 1.0, -3.0, -3.0, 5.0, -1.0, -5.0, 7.0, -5.0, 3.0, 7.0, -7.0};
    float expected_wgrad[out_c * in_c * k * k] = {-33.57872, 4.738071, 1.6974534, -23.931347, 0.0, 0.0, -19.40575, -6.6916566, -20.006197, -35.960274, 0.0, 0.0, -0.0, -3.7966714, -4.6317177, -9.852523, 26.8678, -2.451472, -1.1954706, -0.0, 27.531742, 23.815393, -1.2276684, 7.0083923, 5.3402653, 87.7471, 0.0, 10.364372, -0.0, 0.76612973, 28.654396, 48.899086, -4.414849, 0.0, 0.0, -83.508125, -37.993935, 0.0, -0.85293365, -45.556732, 8.51738, -1.5086063, -0.0, -42.931335, 1.112335, -0.60158986, -0.0, 46.152203, -45.544693, 3.4898016, 5.452224, -29.151382, -57.240128, 0.0, -8.89755, -40.510063, 1.5275986, -8.674077, 0.0, 77.94327, -7.880434, 7.5431643, 0.0, 77.52607, 0.0, -4.222571, 59.362812, 3.8910441, 0.88607216, 59.992985, -3.5048149, -2.120478, -1.9966112, -29.206215, -13.281129, -16.92371, 8.763814, -0.0, 8.043129, -121.80171, -7.076722, 3.5219595, 45.260754, 0.0, 28.612537, -27.613997, 3.1657057, -1.170333, -41.405273, 0.0, -54.34277, 0.0, 0.0, 57.93476, -9.4031105, 0.8015497, 131.81824, -0.0, -0.0, 1.4297582, 16.92097, 60.532425, 0.49837634, -0.0, -34.20508, -1.326139, -6.5906453, -29.209385};
    float expected_dx[batch * in_c * width * height] = {0.0, -2.1782398, 0.0, 0.0, 0.91870284, -34.315372, -4.282489, 6.203444, 1.9730736, 0.0, 0.0, 0.0, 11.319318, 2.5327942, 0.0, -1.6384401, -0.0, 18.292845, -13.652251, -5.9426093, 0.33635727, -14.32013, -0.0, 0.0, 6.0390463, -69.49831, -0.0, -26.248207, -2.0655258, -0.0, -0.0, 0.0, 1.5861868, 23.928194, 8.420558, -10.36971, -0.0, -26.247177, -5.551889, 0.0, 0.0, -0.0, -2.4537346, 13.247859, 0.0, 3.0413444, 10.31106, 6.866543, 0.0, -35.50199, 35.244755, -3.6030715, 7.5782714, -26.472475, -0.0, -6.740296, -1.4607962, -1.4330007, 10.299997, -3.1388197, -5.762588, 0.0, -123.3927, -93.6573, 0.0, 44.529472, -141.99551, 142.95444, -68.35566, 0.0, 73.48724, 20.004158, -3.9685862, -11.278812, 75.30925, 2.285129, -0.0, -0.0, 14.755808, -3.6429567, 0.0, 66.276146, 63.09914, -33.394794, 15.433658, -0.0, -45.18671, 0.0, 22.553936, 11.092219, -2.5287757, 4.2883043, -18.046654, 67.58054, -30.58053, -8.673932, -2.3083377, 17.308912, 0.0, -46.136005, 4.252196, 30.1023, -0.0, -0.0, 22.941467, -0.0, -0.0, -0.10061664, -28.109476, -0.0, -119.48053, 5.560068, 0.0, 6.739938, -0.0, 0.0, -3.4556751, -1.0665026, -0.0, -0.0};
    Halide::Buffer<float> in_view(in, width, height, in_c, batch);
    BConvLayer dut (k, in_c, out_c, 1, true);
    dut.load_weights(W);
    dut.forward_t(out_view, in_view);
    for (int i = 0; i < batch * out_c * out_height * out_width; ++i) {
        if (fabsf(out[i] - expected_out[i]) > 1E-3f) {
            std::cerr << "conv_forward_backward failed at forward_t " << i;
            std::cerr << ". Expected: " << expected_out[i] << " got: " << out[i] << std::endl;
            return -1;
        }
    }
    float ppg [batch * out_c * out_height * out_width];
    for (int i = 0; i < batch*out_width*out_height*out_c; ++i) {
        ppg[i] = (float) i;
    }
    float dx [batch * in_c * height * width];
    Halide::Buffer<float> dx_view(dx, width, height, in_c, batch);
    Halide::Buffer<float> ppg_view(ppg, out_width, out_height, out_c, batch);
    dut.backward(dx_view, ppg_view);
    for (int i = 0; i < out_c * in_c * k * k; ++i) {
        if (fabsf(dut.get_dw()[i] - expected_wgrad[i]) > 1E-3f) {
            std::cerr << "conv_forward_backward failed at backward dw check, ind: " << i;
            std::cerr << ". Expected: " << expected_wgrad[i] << " got: " << dut.get_dw()[i] << std::endl;
            return -1;
        }
    } 
    for (int i = 0; i < batch * in_c * height * width; ++i) {
        if (fabsf(dx[i] - expected_dx[i]) > 1E-3f) {
            std::cerr << "conv_forward_backward failed at backward dx check, ind: " << i;
            std::cerr << ". Expected: " << expected_dx[i] << " got: " << dx[i] << std::endl;
            return -1;
        }
    }
    return 0;
}

int test_save_load_BConvLayer() {
    const int s = 1;
    const int k = 3;
    const int width = 4;
    const int height = 5;
    const int out_height = (height - k) / s + 1;
    const int out_width = (width - k) / s + 1;
    const int in_c = 3;
    const int out_c = 4;
    float out [out_c * out_height * out_width];
    Halide::Buffer<float> out_view(out, out_width, out_height, out_c);
    float in [in_c * width * height] = {
        -1.00865331, -0.29219059,  2.67323418,  1.8020211 ,  0.06334328,
       -0.93614211, -0.0566484 ,  0.10741072, -0.35934454, -2.38577151,
       -1.12968527, -1.73120435, -0.99528351,  0.14717974,  1.10746035,
        0.12275506, -1.23225989, -0.91246136,  0.57759139,  0.14330837,
        0.19972974,  0.23643302,  1.78377793, -1.63716193, -0.71179966,
        0.94430747,  1.31911213, -0.84233116,  0.11643658, -1.69246736,
       -1.64737212,  1.08346985, -0.06631668, -0.62604917,  0.52611927,
       -0.31612989, -1.61702444,  0.99476058, -0.55800325,  2.00426533,
        3.442167  , -1.02430882,  0.09381595, -0.7679083 ,  2.014368  ,
       -0.07990519, -0.20638353,  0.87695587, -2.87695277,  0.63686711,
       -0.37224111,  0.30792107,  0.47311558,  0.53818892, -1.02746744,
       -0.42890086, -0.3766753 ,  0.13679642, -0.32584462,  0.19585104
    };
    float W [k*k*in_c*out_c] = {
        -0.7299722 , -0.78967849,  0.84872669, -0.49856974, -1.09713458,
        1.47446225, -0.48514377, -0.12391957, -0.32268059,  0.34577184,
       -1.07259811,  1.85186512, -1.43826704,  0.37966712,  0.77195291,
        0.08796895, -0.51668848,  0.61286801,  0.59773526, -1.09789223,
       -0.57357792, -0.49615405, -0.61383422, -0.87604905, -0.12136966,
        0.84372212,  2.14358485,  0.30483449,  1.00991064,  0.38306488,
        0.59696656,  0.87319798, -0.06306927, -1.4817166 ,  2.28488271,
       -0.9710247 ,  0.32753392,  1.54709241, -0.1421556 ,  0.52972941,
       -0.85173797,  0.2514344 ,  1.38198671,  0.56488595, -0.27808375,
        0.30079492, -1.5604293 , -0.76920338,  0.94884777, -0.34898017,
       -0.27261119,  0.66253144, -0.44718849, -1.04649192, -0.40443405,
        0.75018633,  0.76379933, -0.18070994,  1.69631008,  0.95052768,
       -0.49252714,  0.09670723,  1.18107807, -0.60567241, -1.61544695,
       -0.70376182, -0.53966195, -0.38910442, -0.14767869, -0.32604882,
        0.03504815,  0.53011946,  0.99830563,  0.2980226 ,  0.18446012,
        0.35257727, -0.39835519,  1.29643823, -0.18279839, -0.80132703,
       -0.13609081,  0.35219595, -0.58026604,  1.25114061,  0.59609455,
       -0.49310708,  0.03367772, -0.29258327, -0.46005861, -2.14134511,
        0.38816264, -2.92246895,  1.87465183, -0.43234896,  0.94031102,
       -0.13359161, -0.59917379, -1.22028055,  1.16533403, -0.71487912,
       -0.13869648, -0.72062408, -0.01038284, -1.82785722,  0.77738807,
        0.03013952, -0.03744685, -0.56171895
    };
    float expected_out[out_c * out_height * out_width] = {13.0, 5.0, -1.0, -7.0, 1.0, 5.0, 5.0, -3.0, -1.0, 5.0, -3.0, -3.0, -7.0, 1.0, -5.0, 5.0, 5.0, 5.0, 3.0, 11.0, 1.0, -1.0, -5.0, 3.0};
    Halide::Buffer<float> in_view(in, width, height, in_c);
    BConvLayer dut (k, in_c, out_c, 1, true);
    dut.load_weights(W);
    std::ofstream fout;
    fout.open("../test_weights/BConvLayerTest.csv", std::ios::out | std::ios::app);
    dut.save_layer(fout);
    fout.close();

    std::ifstream fin;
    fin.open("../test_weights/BConvLayerTest.csv", std::ios::in);
    dut.load_layer(fin);

    for (int x = 0; x < k; ++x) {
        for (int y = 0; y < k; ++y) {
            for (int i = 0; i < in_c; ++i) {
                for (int o = 0; o < out_c; ++o) {
                    float w = W[
                        o * k * k * in_c
                        + i * k * k
                        + y * k
                        + x];
                    float tw =  dut.get_train_w(x, y, i, o);
                    if (w != tw) {
                        std::cerr << "test_save_load_BConvLayer failed at " << x << ", " << y << ", " << i << ", " << o;
                        std::cerr << ". Expected: " << w << ", got: " << tw << std::endl;
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}