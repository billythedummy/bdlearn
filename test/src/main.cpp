#include <bdlearn/bdlearn.hpp>
#include <iostream>

using namespace bdlearn;

int main(int argc, char **argv) {
    BMat orig(9, 9);
    orig.ones();
    BMat copy(orig);
    //std::cout << orig.isEqual(copy) << std::endl;
    std::cout << orig << std::endl;
    std::cout << copy << std::endl;
    BMat res = orig % copy;
    std::cout << res << std:: endl;
    return 0;
}