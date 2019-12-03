#ifndef _BDLEARN_UTILS_H_
#define _BDLEARN_UTILS_H_

#include <fstream>
#include <sstream>
namespace bdlearn {
    void save_arr(float* save_arr, int size, std::fstream& fout);
    void load_arr(float* load_arr, int size, std::fstream& fin);
}
#endif 