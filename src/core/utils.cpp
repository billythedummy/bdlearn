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
    
    int compare_file_output(std::string path1, std::string path2) {
        std::ifstream fin1, fin2;
        fin1.open(path1, std::ios::in);
        fin2.open(path2, std::ios::in);
        std::string line_in, line_out, temp;
        while (fin1 >> temp) {
            getline(fin1, line_in);
            getline(fin2, line_out);
            if (line_in.compare(line_out) != 0) {
                return -1;
            }
        }
        fin1.close();
        fin2.close();
    }
    return 0;
}