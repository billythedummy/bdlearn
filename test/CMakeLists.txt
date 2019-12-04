# target
add_executable(bdlearn_test
    src/main.cpp
    src/BMat_basic.cpp
    src/Halide_test.cpp
    src/BConv_basic.cpp
    src/BatchNorm_basic.cpp
    src/GAP_test.cpp
    src/BatchBlas_test.cpp
    src/Model_test.cpp
    src/Softmax_test.cpp
    src/Ensemble_test.cpp
    src/DataSet_test.cpp
    src/CIFAR_test.cpp
    src/MNIST_test.cpp
    src/MaxPool_test.cpp
)

# link to main lib
target_link_libraries(bdlearn_test
    bdlearn
)

# configure target
target_compile_features(bdlearn_test PRIVATE cxx_auto_type)