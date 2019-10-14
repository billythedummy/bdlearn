#include <string.h>
#include <iostream>
#include <bitset>
#include "bdlearn/BMat.hpp"
#include "libpopcnt.h"

#define PRINT_UNITS 5

namespace bdlearn {

    // Constructors

    BMat::BMat(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        bytes_per_row_ = cols_ / 8;
        if (cols_ % 8) {
            ++bytes_per_row_;
        }
        size_ = rows * cols;
        bytes_ = rows * bytes_per_row_;
        data_.reset(new unsigned char[bytes_]);
        zeros();
    }

    BMat::BMat(const BMat& copy) {
        rows_ = copy.rows_;
        cols_ = copy.cols_;
        bytes_per_row_ = copy.bytes_per_row_;
        size_ = copy.size_;
        bytes_ = copy.bytes_;
        data_.reset(new unsigned char[bytes_]);
        memcpy(data_.get(), copy.data_.get(), bytes_);
    }

    // Destructor

    BMat::~BMat() {
        data_.reset();
    }

    // Public functions

    void BMat::zeros() {
        unsigned char* p = data_.get();
        for (size_t i = 0; i < bytes_; ++i) {
            *p = 0x00;
            ++p;
        }
    }

    void BMat::ones() {
        unsigned char* p = data_.get();
        size_t row_remainder = bytes_per_row_ * 8 - cols_;
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < (bytes_per_row_ - 1); ++j) {
                *p = 0xFF;
                ++p;
            }
            *p = (0xFF << row_remainder);
            ++p;
        }
    }

    // Friend operators

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

    static inline unsigned char make_col_btye(unsigned char* data,
                                                uint8_t bit_index,
                                                size_t bytes_per_row,
                                                uint8_t bits_to_take) {
        // bit_index is from left (MSB)
        unsigned char res = 0x00;
        unsigned char masks[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
        unsigned char mask = masks[bit_index];
        uint8_t bit_index_shift = 7 - bit_index;
        for (uint8_t i = 0; i < (bits_to_take - 1); ++i) {
            res |= (*data & mask) >> bit_index_shift;
            res = res << 1;
            data += bytes_per_row;
        }
        res |= (*data & mask) >> bit_index_shift;
        return res << (8 - bits_to_take);
    }

    void matmul(float* dest, const BMat& A, const BMat& B) {
        // no dimension checking
        // A - m x k, B - k x n, C - m x n
        size_t m, k, n, A_bpr, B_bpr, row_remainder;
        m = A.rows_;
        k = A.cols_; // B.rows_
        n = B.cols_;
        A_bpr = A.bytes_per_row_;
        B_bpr = B.bytes_per_row_;
        row_remainder = A_bpr * 8 - k;
        float dot_sum;
        unsigned char a, b;
        unsigned char* p_a = A.data_.get();
        unsigned char* p_b = B.data_.get();
        unsigned char* p_b_row;
        for (size_t y = 0; y < m; ++y) {
            for (size_t x = 0; x < n; ++x) {
                dot_sum = 0;
                p_b_row = p_b; // keep track of original pointer before vertical advancement
                // Inner loop + edge iterates through yth row of A and xth row of B
                // and computes dot product byte by byte
                for (size_t z = 0; z < (A_bpr - 1); ++z) {
                    a = *p_a;
                    b = make_col_btye(p_b, x % 8, B_bpr, 8);
                    // xnor and popcount
                    b = ~(a ^ b);
                    dot_sum += ((float) (2 * popcnt(&b, 1))) - 8;
                    // advance to next byte of A's row and 8-column of B
                    ++p_a;
                    p_b += 8 * B_bpr;
                }
                // edge case, final byte of row
                a = *p_a;
                b = make_col_btye(p_b, x % 8, B_bpr, 8 - row_remainder);
                // xnor and popcount
                b = ~(a ^ b);
                dot_sum += ((float) (2 * popcnt(&b, 1))) - 8 - row_remainder;

                // update entry of dest
                dest[y*n + x] = dot_sum;
                p_a -= A_bpr - 1; // reset to start of row of A for next col of B
                // reset to top of matrix
                p_b = p_b_row;
                if (x % 8 == 7) { // advance to next byte to build next col of B
                    ++p_b;
                }
            }
            p_a += A_bpr; // advance to next row of A
            p_b = B.data_.get(); // reset to 0th column for B
        }
    }
}