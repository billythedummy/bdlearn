#include <bdlearn/bdlearn.hpp>
#include <iostream>

using namespace bdlearn;

int main(int argc, char **argv) {
    BMat orig(2, 4);
    orig.ones();
    BMat copy(orig);
    //std::cout << (orig == copy) << std::endl;
    std::cout << orig << std::endl;
    //std::cout << copy << std::endl;
    return 0;
}