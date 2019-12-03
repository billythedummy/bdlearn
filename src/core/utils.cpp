#include "bdlearn/utils.hpp"

namespace bdlearn {
    void save_arr(float* save_arr, int size, std::ofstream& fout) {
        for (int i = 0; i < size; ++i) {
            fout << save_arr[i] << ",";
        }
        fout << std::endl;
    }
    void load_arr(float* load_arr, int size, std::ifstream& fin) {
        std::string line, data;
        std::string::size_type sz;
        getline(fin, line);
        std::istringstream s(line);
        for (int i = 0; i < size; ++i) {
            std::getline(s, data, ',');
            load_arr[i] = std::stof(data, &sz);
        }
    }
}