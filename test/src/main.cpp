#include <bdlearn/bdlearn.hpp>
#include <iostream>

int main(int argc, char **argv) {
    bdlearn::BMat orig(12, 11);
    orig.ones();
    bdlearn::BMat copy(orig);
    //std::cout << orig.isEqual(copy) << std::endl;
    std::cout << orig << std::endl;
    std::cout << copy << std::endl;
    return 0;
}