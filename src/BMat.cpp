#include <string.h>
#include "bdlearn/BMat.hpp"

namespace bdlearn {
    BMat::BMat(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        size_ = rows * cols;
        bytes_ = size_ / 8;
        if (size_ % 8) { // not divisible by 8
            bytes_++;
        } 
        data_ = new char[bytes_];
    }

    BMat::BMat(const BMat& copy) {
        rows_ = copy.rows_;
        cols_ = copy.cols_;
        size_ = copy.size_;
        bytes_ = copy.bytes_;
        data_ = new char[bytes_];
        memcpy(data_, copy.data_, bytes_);
    }

    BMat::~BMat() {
        delete[] data_;
    }

    bool BMat::IsEqual(const BMat& comp) {
        if (rows_ != comp.rows_ || cols_ != comp.cols_) {
            return false;
        }
        for (size_t i = 0; i < bytes_; i++) {
            if (data_[i] != comp.data_[i]) {
                return false;
            }
        }
        return true;
    }
}