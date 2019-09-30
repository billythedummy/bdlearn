#ifndef _BDLEARN_BMAT_H_
#define _BDLEARN_BMAT_H_

#include <cstddef>

namespace bdlearn {
    class BMat {
        public:
        // Constructors
            // default - uninitialized memory
            BMat(size_t rows, size_t cols);
            // copy
            BMat(const BMat& copy);

        // Destructors
            virtual ~BMat();

        // public functions
            bool IsEqual(const BMat& comp);

        private:
            char* data_;
            size_t rows_;
            size_t cols_;
            size_t size_;
            size_t bytes_;
    };
}

#endif