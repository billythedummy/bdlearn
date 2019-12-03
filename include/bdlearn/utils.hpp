#ifndef _BDLEARN_UTILS_H_
#define _BDLEARN_UTILS_H_

#include <fstream>

namespace bdlearn {
    void save_arr(float* arr, int size, std::fstream fout);
    void load_arr(float* arr, int size, std::fstream fin);
}

#endif 