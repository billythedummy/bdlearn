#ifndef _BDLEARN_BATCHDATA_H_
#define _BDLEARN_BATCHDATA_H_

namespace bdlearn {
    struct batchdata {
        float* x_ptr;
        float* y_ptr;
        int size;
    };
    inline void free_batch_data(batchdata b) {delete[] b.x_ptr; delete[] b.y_ptr;}
}

#endif