import numpy as np

class SoftmaxCrossEntropyLossLayer():
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        # q, one_hot and targets from last forward computation
        self.q = None
        self.one_hot = None
        self.targets = None

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: N-Dimensional non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets.
            Example: inputs might be (4 x 10), targets (4) and axis 1.
        :param targets: (N-1)-Dimensional class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO
        self.targets = targets
        max_over_dims = np.amax(logits, axis=axis, keepdims=True) #(n x 1)
        logits -= max_over_dims
        exp_logits = np.exp(logits) # (n x D)
        sum_exp = np.sum(exp_logits, axis=axis, keepdims=True) #(n x 1)
        self.q = exp_logits / sum_exp # (n x D)
        log_sum_exp = np.log(sum_exp)
        log_q = logits - log_sum_exp #(n x D)

        self.one_hot = self._one_hot_encode(logits, targets, axis=axis)
        total_cross_entropy = self.one_hot * -log_q
        total_cross_entropy = np.sum(total_cross_entropy, axis=axis)

        if self.reduction == "mean":
            return np.mean(total_cross_entropy)
        elif self.reduction == "sum":
            return np.sum(total_cross_entropy)
        return 0

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        res = self.q - self.one_hot
        if self.reduction == "sum":
            return res
        elif self.reduction == "mean":
            return res / (self.targets.size)

    def _one_hot_encode(self, logits, targets, axis=-1):
        # dimension oblivious one-hot means dimension oblivious softmax
        idx = np.unravel_index(np.arange(targets.size), targets.shape)
        lidx = list(idx)
        if axis < 0:
            axis = len(logits.shape) + axis
        lidx.insert(axis, targets.flatten())
        lidx = np.array(lidx)
        res = np.zeros_like(logits)
        res[tuple(lidx)] = 1
        return res

if __name__ == "__main__":
    batch = 3
    classes = 10
    truth = np.array([8, 7, 0])
    fx = (np.array([-1.34340341, -0.45974003, -1.13929274, -0.70190898, -0.44352927,
        0.75805997,  0.57303502, -0.48544892, -0.84611373,  2.63468655,
       -0.89735092, -0.68660326, -0.17047781,  0.71524846, -1.53322612,
        1.51689624, -1.01533733, -0.34391119, -0.48589473, -0.11330478,
        0.39868253, -1.51340956,  0.66346402,  0.68727878,  2.15907718,
        1.19633974,  0.46431238, -1.61770077, -0.02563707,  0.70544018])
        .astype(np.float32)
        .reshape((batch, classes)))
    dut = SoftmaxCrossEntropyLossLayer()
    print(dut.forward(fx, truth))
    print(list(dut.q.flatten()))
    print(list(dut.backward().flatten()))