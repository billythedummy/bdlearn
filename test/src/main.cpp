#include <bdlearn/bdlearn.hpp>
#include <iostream>

int main(int argc, char **argv) {
    bdlearn::BMat orig(1, 1);
    bdlearn::BMat copy(orig);
    std::cout << orig.IsEqual(copy) << std::endl;
    //std::cout << "fuck" << std::endl;
    return 0;
}