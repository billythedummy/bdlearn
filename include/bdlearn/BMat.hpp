#ifndef _BDLEARN_BMAT_H_
#define _BDLEARN_BMAT_H_

#include <cstddef>
#include <iostream>
#include "Halide.h"

namespace bdlearn {
    class BMat {
        public:
        // Constructors
            // default dummy
            BMat();
            // zero initialized
            BMat(size_t rows, size_t cols);
            // from sign(src)
            BMat(size_t rows, size_t cols, float* src);
            // copy
            BMat(const BMat& copy);

        // Destructor
            virtual ~BMat();

        // public functions
            void zeros();
            void ones();
            void random();
            void sign(float *src);
            size_t rows() const;
            size_t cols() const;
            uint8_t get(int row, int col) const;

        // friend operators
        friend bool operator==(const BMat& a, const BMat& b);
        friend std::ostream& operator<<(std::ostream& os, const BMat& bmat);
        // C += A @ B
        friend void matmul(Halide::Buffer<float>* dest, const BMat& A, const BMat& B);

        private:
            std::unique_ptr<uint8_t[]> data_;
            size_t rows_;
            size_t cols_;
            size_t size_; // rows * cols

            BMat& operator=(const BMat& ref) = delete;
    };
}

#endif