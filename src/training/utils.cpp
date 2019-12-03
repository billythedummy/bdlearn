#include "bdlearn/utils.hpp"

namespace bdlearn {
    void save_arr(float* save_arr, int size, std::fstream& fout) {
        std::ostringstream oss;
        for (int i = 0; i < size; ++i) {
            oss << save_arr[i] << ",";
            //fout << train_w_.get()[i] << ",";
        }
        oss << "\n";
        fout << oss.rdbuf() << std::endl;
    }
    void load_arr(float* load_arr, int size, std::fstream& fin) {

    }
}