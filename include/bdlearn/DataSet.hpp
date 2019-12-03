#ifndef _BDLEARN_DATASET_H_
#define _BDLEARN_DATASET_H_

#include <memory>
#include <string.h> //memcpy
#include "darknet_image_loader.h"
#include "bdlearn/BufDims.hpp"
#include "bdlearn/BatchData.hpp"

namespace bdlearn {
    // ONLY HANDLES CLASSIFICATION ONE-HOT DATA FOR NOW
    class DataSet {
        public:
        // constructor
            DataSet(){};

        // destructor
            ~DataSet(){};

        // public functions
            void load_darknet_classification(char* images, char* label_file);
            void shuffle(void);
            batchdata get_next_batch(void);

            // getter setters
            void set_batch_size(int batch_size);
            int get_classes(void) {return classes_;}
            int get_epoch_size(void) {return epoch_size_;}
            int get_steps(void) {return steps_;}
            int get_curr_step(void) {return curr_step_;}
            int* get_train_i(void) {return train_i_.get();}
            bufdims get_x_dims(void) {return x_dims_;}

        private:
            // delete assigment
            DataSet& operator=(const DataSet& ref) = delete;

        // private functions
            batchdata get_next_batch_normal(void);
            batchdata get_next_batch_rem(void);
        // private fields
            int batch_size_;
            int epoch_size_;
            int steps_; // + 1 if remainder
            int curr_step_;
            bufdims x_dims_;
            int x_size_; // x_dims_.w * .h *.c
            int classes_; // DataSet only handles classification data for now
            std::unique_ptr<float[]> X_;
            std::unique_ptr<float[]> Y_;
            std::unique_ptr<int[]> train_i_; // order of iteration for training this epoch
    };
}

#endif