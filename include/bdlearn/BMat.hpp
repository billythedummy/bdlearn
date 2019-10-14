#ifndef _BDLEARN_BMAT_H_
#define _BDLEARN_BMAT_H_

#include <cstddef>
#include <iostream>
#include <memory>

namespace bdlearn {
    // All rows of BMats are byte-aligned, so each row is zero-padded out on the right
    class BMat {
        public:
        // Constructors
            // default - zero initialized
            BMat(size_t rows, size_t cols);
            // copy
            BMat(const BMat& copy);

        // Destructor
            virtual ~BMat();

        // public functions
            void zeros();
            void ones();
        
        // friend operators
        friend bool operator==(const BMat& a, const BMat& b);
        friend std::ostream& operator<<(std::ostream& os, const BMat& bmat);
        // C += A @ B
        friend void matmul(float* dest, const BMat& A, const BMat& B);


        private:
            std::unique_ptr<unsigned char[]> data_;
            size_t rows_;
            size_t cols_;
            size_t bytes_per_row_;
            size_t size_; // rows * cols
            size_t bytes_; // rows * bytes_per_row
            BMat& operator=(const BMat& ref) = delete;
    };
}

#endif