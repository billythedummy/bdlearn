cmake_minimum_required(VERSION 3.5)
project(bdlearn VERSION 0.0.0 LANGUAGES CXX)

# variables

# create target
add_library(bdlearn
    SHARED
    src/core/BMat.cpp
)

# dependencies
target_link_libraries(bdlearn
    Halide
)

# Python bindings
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/pybind11)
pybind11_add_module(bdlearnpy
    src/python/py_bindings.cpp
    src/core/BMat.cpp
)

# include
target_include_directories(bdlearn
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(bdlearnpy
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# target
target_compile_features(bdlearn PRIVATE cxx_auto_type)
target_compile_options(bdlearn PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall>)

# test
add_subdirectory(test)
