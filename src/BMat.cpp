#include <string.h>
#include "bdlearn/BMat.hpp"
#include <iostream>

#define PRINT_UNITS 5

namespace bdlearn {
    BMat::BMat(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        size_ = rows * cols;
        bytes_ = size_ / 8;
        if (size_ % 8) { // not divisible by 8
            ++bytes_;
        } 
        data_ = new unsigned char[bytes_];
    }

    BMat::BMat(const BMat& copy) {
        rows_ = copy.rows_;
        cols_ = copy.cols_;
        size_ = copy.size_;
        bytes_ = copy.bytes_;
        data_ = new unsigned char[bytes_];
        memcpy(data_, copy.data_, bytes_);
    }

    BMat::~BMat() {
        delete[] data_;
    }

    bool BMat::isEqual(const BMat& comp) {
        if (rows_ != comp.rows_ || cols_ != comp.cols_) {
            return false;
        }
        for (size_t i = 0; i < bytes_; ++i) {
            if (data_[i] != comp.data_[i]) {
                return false;
            }
        }
        return true;
    }

    void BMat::zeros() {
        for (size_t i = 0; i < bytes_; ++i) {
            *(data_ + i) = 0x00;
        }
    }

    void BMat::ones() {
        for (size_t i = 0; i < bytes_; ++i) {
            *(data_ + i) = 0xFF;
        }
    }

    BMat operator%(const BMat& a, const BMat& b) {
        // naive implementation
        BMat temp(a.rows_, b.cols_);
        return temp;
    }

    std::ostream& operator<<(std::ostream& os, const BMat& bmat) { 
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
                        size_t bit_index = r * bmat.cols_ + c;
                        size_t byte_index = bit_index / 8;
                        size_t offset = bit_index % 8;
                        unsigned char mask = 0x80 >> offset;
                        unsigned char printed = (bmat.data_[byte_index] & mask) >> (7 - offset);
                        os << " " << (unsigned int) printed << " ";
                    }
                }
            }
            os << std::endl;
        }
        return os;
    }
}