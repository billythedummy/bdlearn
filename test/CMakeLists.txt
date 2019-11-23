# target
add_executable(bdlearn_test
    src/main.cpp
    src/BMat_basic.cpp
    src/Halide_test.cpp
    src/BConv_basic.cpp
)

# link to main lib
target_link_libraries(bdlearn_test bdlearn)

# configure target
target_compile_features(bdlearn_test PRIVATE cxx_auto_type)