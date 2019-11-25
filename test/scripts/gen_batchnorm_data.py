from torch import nn
from torch import tensor
import torch

n = 3
c = 2
h = 7
w = 5
gamma = [0.45, 0.73]
beta = [0.12, 0.3419]
in_data = [0.0747, -1.0128, -1.5366, -0.6547, -0.0991, -0.9915, -0.5302,  1.4154,
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
        0.4695, -0.2552]

weight = nn.Parameter(tensor(gamma, dtype=torch.float32))
bias = nn.Parameter(tensor(beta, dtype=torch.float32))
bn = nn.BatchNorm2d(c)
bn.weight = weight
bn.bias = bias
test_tensor = tensor(in_data, dtype=torch.float32).reshape((n, c, h, w))
print(bn(test_tensor).flatten())
print(bn.running_mean)
print(bn.running_var)
