#ifndef _BDLEARN_UTILS_H_
#define _BDLEARN_UTILS_H_

#include <fstream>
#include <sstream>
namespace bdlearn {
    void save_arr(float* save_arr, int size, std::ofstream& fout);
    void load_arr(float* load_arr, int size, std::ifstream& fin);
    int compare_file_output(std::string path1, std::string path2);
}
#endif 