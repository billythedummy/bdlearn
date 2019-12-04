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
        std::string line1, line2, temp;
        while (fin1 >> line1) {
            fin2 >> temp;
            getline(fin2, line2);
            if (line1.compare(line2) != 0) {
                std::cerr << temp << std::endl;
                std::cerr << line1 << std::endl;
                std::cerr << line2 << std::endl;
                return -1;
            }
        }
        fin1.close();
        fin2.close();
        return 0;
    }
}