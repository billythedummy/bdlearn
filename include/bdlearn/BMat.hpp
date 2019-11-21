#ifndef _BDLEARN_BMAT_H_
#define _BDLEARN_BMAT_H_

#include <cstddef>
#include <iostream>
#include <memory>
#include "Halide.h"

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
            void random();
            size_t rows() const;
            size_t cols() const;
            uint8_t get(int row, int col) const;

        // friend operators
        friend bool operator==(const BMat& a, const BMat& b);
        friend std::ostream& operator<<(std::ostream& os, const BMat& bmat);
        // C += A @ B
        friend void matmul(Halide::Buffer<float>* dest, const BMat& A, const BMat& B);
        template<typename T>
        friend void im2col(
            Halide::Buffer<T>& src,
            size_t w_src, size_t h_src, size_t c_src,
            size_t p_x, size_t p_y,
            size_t s_x, size_t s_y,
            size_t k_x, size_t k_y,
            Halide::Buffer<T>* dest
        ) {
            const int out_height = (im.h + 2*p_y - k_y) / s_y + 1;
            const int out_width = (im.w + 2*p_x - k_x) / s_x + 1;
            const int patch_area = k_x * k_y;
            const int h_im2col = patch_area * im.c;
            const int w_im2col = out_height * out_width;
            Halide::Func bim2col("bim2col");
            Halide::Var x, y;
            const int c = y / patch_area;
            const int pix_index_in_patch = x % patch_area;
            const int which_row = x / out_width;
            const int which_patch_in_row = x % out_width;
            const int top_left_y_index = which_row * s_y - p_y;
            const int top_left_x_index = which_patch_in_row * s_x - p_x;
            const int y_index = top_left_y_index + (pix_index_in_patch / k_x);
            const int x_index = top_left_x_index + (pix_index_in_patch % k_x);
            bim2col(x, y) = src.get()[c*h_src*w_src+ y_index*w_src + x_index]
            
            Halide::Func out;
            out(x, y) = bim2col(x, y);
            // Scheudle
            // TO-DO OPTIMIZE ALL THE FANCY STUFF
            out.realize(*dest);
        }
        private:
            std::unique_ptr<uint8_t[]> data_;
            size_t rows_;
            size_t cols_;
            size_t size_; // rows * cols
            BMat& operator=(const BMat& ref) = delete;
    };
}

#endif