#include <string.h>
#include <iostream>
#include "bdlearn/BMat.hpp"

#define PRINT_UNITS 5

namespace bdlearn {

    // Constructors

    BMat::BMat(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        size_ = rows * cols;
        data_.reset(new uint8_t[size_]);
        zeros();
    }

    BMat::BMat(size_t rows, size_t cols, float* src) {
        rows_ = rows;
        cols_ = cols;
        size_ = rows * cols;
        data_.reset(new uint8_t[size_]);
        Halide::Buffer<uint8_t> data_view(data_.get(), size_);
        Halide::Buffer<float> src_view(src, size_);
        Halide::Var i;
        // Algo
        Halide::Func sign;
        sign(i) = Halide::cast<uint8_t>(src_view(i) >= 0);
        sign.realize(data_view);
    }

    BMat::BMat(const BMat& copy) {
        rows_ = copy.rows_;
        cols_ = copy.cols_;
        size_ = copy.size_;
        data_.reset(new uint8_t[size_]);
        memcpy(data_.get(), copy.data_.get(), size_);
    }

    // Destructor

    BMat::~BMat() {
        data_.reset();
    }

    // Public functions

    void BMat::zeros() {
        // Vars
        Halide::Buffer<uint8_t> view(data_.get(), size_, "view");
        Halide::Var i;
        // Algo
        Halide::Func set;
        set(i) = Halide::cast<uint8_t>(0x00);
        // Schedule
        // TO-DO: Optimize
        set.realize(view);
    }

    void BMat::ones() {
        // Vars
        Halide::Buffer<uint8_t> view(data_.get(), size_, "view");
        Halide::Var i;
        // Algo
        Halide::Func set;
        set(i) = Halide::cast<uint8_t>(0x01);
        // Schedule
        // TO-DO: Optimize
        set.realize(view);
    }

    void BMat::random() {
        // Vars
        Halide::Buffer<uint8_t> view(data_.get(), size_, "view");
        Halide::Var i;
        // Algo
        Halide::Expr seed = (int) time(NULL);
        Halide::Expr rand = (Halide::random_uint(seed) / 10) % 2;
        Halide::Func set;
        set(i) = Halide::cast<uint8_t>(rand);
        // Schedule
        // TO-DO: Optimize
        set.realize(view);
    }

    void BMat::sign(float *src) {
        Halide::Buffer<uint8_t> data_view(data_.get(), size_);
        Halide::Buffer<float> src_view(src, size_);
        Halide::Var i;
        // Algo
        Halide::Func sign;
        sign(i) = Halide::cast<uint8_t>(src_view(i) >= 0);
        sign.realize(data_view);
    }

    size_t BMat::rows() const {
        return rows_;
    }

    size_t BMat::cols() const {
        return cols_;
    }

    uint8_t BMat::get(int row, int col) const {
        return data_.get()[row * cols_ + col];
    }

    // Friend operators

    bool operator==(const BMat& a, const BMat& b) {
        if (a.rows_ != b.rows_ || a.cols_ != b.cols_) {
            return false;
        }
        for (size_t i = 0; i < a.size_; ++i) {
            if (a.data_[i] != b.data_[i]) {
                return false;
            }
        }
        return true;
    }

    std::ostream& operator<<(std::ostream& os, const BMat& bmat) {
        os << std::hex;
        for (size_t r = 0; r < bmat.rows_; ++r) {
            size_t bot_margin = bmat.rows_ - PRINT_UNITS - 1;
            if (r > PRINT_UNITS - 1 && bmat.rows_ > 2*PRINT_UNITS && r < bot_margin) {
                for (size_t i = 0; i < 3; ++i) {
                    os << "   ." << std::endl;
                }
                r = bot_margin;
            } else {
                for (size_t c = 0; c < bmat.cols_; ++c) {
                    size_t right_margin = bmat.cols_ - PRINT_UNITS - 1;
                    if (c > PRINT_UNITS - 1 && bmat.cols_ > 2*PRINT_UNITS && c < right_margin) {
                        os << " . . . ";
                        c = right_margin;
                    } else {
                        size_t row_bytes_traversed = r * bmat.cols_;
                        size_t byte_index = row_bytes_traversed + c;
                        os << " " << bmat.data_[byte_index] + 0 << " ";
                    }
                }
            }
            os << std::endl;
        }
        os << std::dec;
        return os;
    }

    void matmul(Halide::Buffer<float>* dest, const BMat& A, const BMat& B) {
        // Cols first then rows
        // A - m x k, B - k x n, C - m x n
        // C = A @ B
        // Vars
        assert(A.cols_ == B.rows_);
        Halide::Func bmatmul("bmatmul");
        Halide::Var x, y;
        Halide::Buffer<uint8_t> A_buf(A.data_.get(), A.cols_, A.rows_, "A_buf");
        Halide::Buffer<uint8_t> B_buf(B.data_.get(), B.cols_, B.rows_, "B_buf");
        const int k_size = A.cols_;
        Halide::RDom k(0, k_size);
        // Algo
        Halide::Expr xnor = ~( A_buf(k, y) ^ B_buf(x, k) );
        Halide::Expr popcnt = Halide::cast<float>(Halide::popcount(xnor));
        bmatmul(x, y) += 2 * popcnt - 15;
        // This proxy func will be used for optimizations later
        Halide::Func out;
        out(x, y) = bmatmul(x, y);
        // Scheudle
        // TO-DO OPTIMIZE ALL THE FANCY STUFF
        out.realize(*dest);
        return;
    }
}