#include "bdlearn/DataSet.hpp"

namespace bdlearn {
    
    // CODE COPIED FROM darknet/uwimg
    // for use under the 
    // DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
    // All credit to https://github.com/pjreddie
    void DataSet::load_darknet_classification(char* images, char* label_file) {
        dn_list *image_list = get_lines(images);
        dn_list *label_list = get_lines(label_file);
        int k = label_list->size;
        char **labels = (char **)list_to_array(label_list);

        int n = image_list->size;
        dn_node *nd = image_list->front;

        float* X;
        float* Y;
        int* train_i;

        int i;
        int count = 0;
        bool init = false;
        while(nd){
            char *path = (char *)nd->val;
            dn_image im = load_image(path);
            if (!init) {
                x_dims_ = {.w=im.w, .h=im.h, .c=im.c};
                x_size_ = im.w*im.h*im.c;
                classes_ = k;
                epoch_size_ = n;
                if (batch_size_ != 0) {
                    set_batch_size(batch_size_); // to get steps correct
                }
                X = new float[x_size_*n]();
                Y = new float[k*n]();
                train_i = new int[n];
                init = true;
            }
            train_i[count] = count;
            memcpy(X+count*x_size_, im.data, x_size_*sizeof(float));
            for (i = 0; i < k; ++i) {
                int y_i = count * classes_;
                if (strstr(path, labels[i])) {
                    Y[y_i + i] = 1;
                }
            }
            ++count;
            nd = nd->next;
        }
        free_list(image_list);
        X_.reset(X);
        Y_.reset(Y);
        train_i_.reset(train_i);
    }

    void DataSet::shuffle() {
        // Fisher-yates shuffle train_i_
        for (int i = 0; i < epoch_size_-1; ++i) {
            int swap_index = rand() % (epoch_size_-i) + i;
            int swap = train_i_[swap_index];
            train_i_[swap_index] = train_i_[i];
            train_i_[i] = swap;
        }
    }

    batchdata DataSet::get_next_batch() {
        batchdata res;
        if (curr_step_ == steps_ - 1 && epoch_size_ % batch_size_) {
            res = get_next_batch_rem();
        } else {
            res = get_next_batch_normal();
        }
        curr_step_ = (curr_step_ + 1) % steps_;
        return res;
    }

    void DataSet::set_batch_size(int batch_size) {
        batch_size_ = batch_size;
        if (epoch_size_ != 0) {
            steps_ = epoch_size_ / batch_size_;
            if (epoch_size_ % batch_size_) {
                steps_++;
            }
        }
    }

    // private functions

    batchdata DataSet::get_next_batch_normal() {
        int batch_offset = curr_step_ * batch_size_;
        float* x_buffer = new float[batch_size_];
        float* y_buffer = new float[batch_size_];
        for (int j = 0; j < batch_size_; ++j) {
            int curr_index = batch_offset + j;
            int data_index = train_i_[curr_index];
            memcpy(x_buffer + j*x_size_, X_.get()+data_index*x_size_,
                    sizeof(float) * x_size_);
            memcpy(y_buffer + j*classes_, Y_.get()+data_index*classes_,
                    sizeof(float) * classes_);
        }
        return {.x_ptr=x_buffer, .y_ptr=y_buffer, .size=batch_size_};
    }

    batchdata DataSet::get_next_batch_rem() {
        const int rem = epoch_size_ % batch_size_;
        float* x_buffer = new float[batch_size_];
        float* y_buffer = new float[batch_size_];
        for (int j = (epoch_size_ - rem); j < epoch_size_; ++j) {
            int j_zero_relative = j - epoch_size_ + rem;
            int data_index = train_i_[j];
            memcpy(x_buffer + j_zero_relative*x_size_, X_.get()+data_index*x_size_,
                    sizeof(float) * x_size_);
            memcpy(y_buffer + j_zero_relative*classes_, Y_.get()+data_index*classes_,
                    sizeof(float) * classes_);
        }
        return {.x_ptr=x_buffer, .y_ptr=y_buffer, .size=rem};
    }
}