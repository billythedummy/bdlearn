import numpy as np

def im2col_helper(col, data, p, s, k, out_width):
    patch_area = k * k
    for n in range(col.shape[0]):
        for i in range(col.shape[1]):
            for j in range(col.shape[2]):
                c = i // patch_area
                pix_index_in_patch = i % patch_area
                which_row = j // out_width
                which_patch_in_row = j % out_width
                top_left_y_index = which_row * s - p
                top_left_x_index = which_patch_in_row * s - p
                y_index = top_left_y_index + (pix_index_in_patch // k)
                x_index = top_left_x_index + (pix_index_in_patch % k)
                oob = (y_index < 0 or y_index >= data.shape[2]
                        or x_index < 0 or x_index >= data.shape[3])
                col[n, i, j] = 0 if oob else data[n, c, y_index, x_index]
    return col

def im2col(data, p, s, k, out_width, out_height):
    res = np.empty((data.shape[0],
                    k*k*data.shape[1],
                    out_height * out_width),
                    dtype=np.float32) # N x i2cH x i2cW
    return im2col_helper(res, data, p, s, k, out_width)

def col2im_accum_helper(im, col, p, s, k, out_width, out_height):
    # padding, stride, kernel size
    # col is N x kkc x nPatches
    # im is N x C x h x w, zeros
    h = im.shape[2]
    w = im.shape[3]
    patch_area = k * k
    patches_per_row = out_width
    n_patch_rows = out_height
    for n in range(im.shape[0]):
        for c in range(im.shape[1]):
            for i in range(h):
                for j in range(w):
                    # neighborhood index: 0 - top-left, k^2 - 1 - bot right
                    for nb_index in range(patch_area):
                        # row_index is function of input_channel and nb_index
                        row_index = c * patch_area
                        row_index += nb_index
                        # col_index is which patch # this pixel belonged to
                        top_left_i = i - (nb_index // k)
                        top_left_j = j - (nb_index % k)
                        # Determine row index and col index of patch based on
                        # top left coordinate of patch
                        which_patch_row = (top_left_i + p) // s
                        invalid = (top_left_i + p) % s
                        which_patch_in_row = (top_left_j + p) // s
                        invalid |= (top_left_j + p) % s
                        # Check if patch index is invalid
                        invalid |= which_patch_in_row < 0 or which_patch_in_row >= patches_per_row
                        invalid |= which_patch_row < 0 or which_patch_row >= n_patch_rows
                        # Get final patch index
                        which_patch = which_patch_row * patches_per_row + which_patch_in_row
                        im[n, c, i, j] += 0 if invalid else col[n, row_index, which_patch]
    return im

def col2im_accum(col, in_h, in_w, out_h, out_w, outC, inC, p, s, k):
    # kkc = kernel_size * kernel_size * COld
    # col is N x kkc x nPatches
    # returns N x COld x im_height x im_width
    im = np.zeros(
        (col.shape[0],
        inC, #COld
        in_h,
        in_w),
        dtype=np.float32
    )
    return col2im_accum_helper(im, col,
                                    p, s, k,
                                    out_w, out_h)

def forward(data, W, p, s, k):
    # W should already be in im2col format
    # Save state vars for backwards
    #self.out_height = out_height
    #self.out_width = out_width
    #self.im_height = data.shape[2]
    #self.im_width = data.shape[3]
    # im2col, f2rows
    out_height = (data.shape[2] + 2*p - k) // s + 1
    out_width = (data.shape[3] + 2*p - k) // s + 1
    col_in = im2col(data, p, s, k, out_width, out_height)
    #self.prev_cols = col_in
    #print(W.size - np.count_nonzero(np.sign(W)))
    #print(col_in.size - np.count_nonzero(col_in))
    col = np.sign(W) @ np.sign(col_in)
    return col.reshape(col.shape[0], col.shape[1], out_height, out_width), np.sign(col_in)

def backward(previous_partial_gradient, prev_cols, w, outC, inC, k,
                in_h, in_w, out_h, out_w, p, s, x):
    # w is outC, kkinC
    previous_partial_gradient = previous_partial_gradient.reshape(previous_partial_gradient.shape[0],
                                    previous_partial_gradient.shape[1], -1)
    # Update our weight gradients
    weight_grad_rows = previous_partial_gradient @ np.swapaxes(prev_cols, 1, 2)
    weight_grad_rows = np.sum(weight_grad_rows, axis=0)
    weight_grad = weight_grad_rows.reshape(outC, inC, k, k)
    w_ste = w.copy().reshape(outC, inC, k, k)
    w_ste[np.abs(w_ste) >= 1] = 0
    weight_grad *= w_ste
    # Pass the gradients on to layer behind
    col = np.expand_dims(w.T, 0) @ previous_partial_gradient
    dx = col2im_accum(col, in_h, in_w, out_h, out_w, outC, inC, p, s, k)
    x_ste = x.copy()
    x_ste[np.abs(x_ste) >= 1] = 0
    dx *= x_ste
    return weight_grad, dx

if __name__ == "__main__":
    p = 0
    s = 1
    k = 3
    width = 4
    height = 5
    inC = 3
    outC = 4
    batch = 2
    out_height = (height + 2*p - k) // s + 1
    out_width = (width + 2*p - k) // s + 1
    IN = [
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
    ]
    # to allow for easy changing of batch size   
    IN = np.array(IN[:width*height*inC*batch]).astype(np.float32).reshape(batch, inC, height, width)
    W = np.array(
        [-0.7299722 , -0.78967849,  0.84872669, -0.49856974, -1.09713458,
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
        0.03013952, -0.03744685, -0.56171895]
    ).astype(np.float32).reshape(outC, inC*k*k)
    #i2c = im2col(IN, p, s, k)
    #print(i2c)
    out, col_in = forward(IN, W, p, s, k)
    print("out")
    print(list(out.flatten()))
    ppg = np.arange(batch*outC*out_height*out_width).reshape(batch, outC, out_height, out_width).astype(np.float32)
    weight_grad, dx = backward(ppg, col_in, W, outC, inC, k,
                                height, width, out_height, out_width, p, s, IN)
    #print("dx")
    print(list(dx.flatten()))
    #print("weight_grad") 
    print(list(weight_grad.flatten()))
