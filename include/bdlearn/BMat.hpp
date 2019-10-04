#ifndef _BDLEARN_BMAT_H_
#define _BDLEARN_BMAT_H_

#include <cstddef>
#include <iostream>

namespace bdlearn {
    // All rows of BMats are byte-aligned, so each row is padded out on the right
    class BMat {
        public:
        // Constructors
            // default - uninitialized memory
            BMat(size_t rows, size_t cols);
            // copy
            BMat(const BMat& copy);

        // Destructor
            virtual ~BMat();

        // public functions
            void zeros();
            void ones();
        
        // friend operators
        // Mat mul
        friend bool operator==(const BMat& a, const BMat& b);
        friend std::ostream& operator<<(std::ostream& os, const BMat& bmat);


        private:
            unsigned char* data_;
            size_t rows_;
            size_t cols_;
            size_t bytes_per_row_;
            size_t size_;
            size_t bytes_;
            BMat& operator=(const BMat& ref) = delete;
    };
}

#endif