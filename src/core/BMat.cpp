#include <string.h>
#include <iostream>
#include "bdlearn/BMat.hpp"
#include "libpopcnt.h"

#define PRINT_UNITS 5

namespace bdlearn {
    BMat::BMat(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        bytes_per_row_ = cols_ / 8;
        if (cols_ % 8) {
            ++bytes_per_row_;
        }
        size_ = rows * cols;
        bytes_ = rows * bytes_per_row_;
        data_ = new unsigned char[bytes_];
    }

    BMat::BMat(const BMat& copy) {
        rows_ = copy.rows_;
        cols_ = copy.cols_;
        bytes_per_row_ = copy.bytes_per_row_;
        size_ = copy.size_;
        bytes_ = copy.bytes_;
        data_ = new unsigned char[bytes_];
        memcpy(data_, copy.data_, bytes_);
    }

    BMat::~BMat() {
        delete[] data_;
    }

    bool operator==(const BMat& a, const BMat& b) {
        if (a.rows_ != b.rows_ || a.cols_ != b.cols_) {
            return false;
        }
        for (size_t i = 0; i < a.bytes_; ++i) {
            if (a.data_[i] != b.data_[i]) {
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
                        size_t row_bytes_traversed = r * bmat.bytes_per_row_;
                        size_t col_bytes_traversed = c / 8;
                        size_t offset_in_byte = c % 8;
                        size_t byte_index = row_bytes_traversed + col_bytes_traversed;
                        unsigned char mask = 0x80 >> offset_in_byte;
                        unsigned char printed = (bmat.data_[byte_index] & mask) >> (7 - offset_in_byte);
                        os << " " << (unsigned int) printed << " ";
                    }
                }
            }
            os << std::endl;
        }
        return os;
    }
}