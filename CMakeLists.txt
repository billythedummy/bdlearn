cmake_minimum_required(VERSION 3.5)
project(bdlearn VERSION 0.0.0 LANGUAGES CXX)

# dependencies

# create target
add_library(bdlearn
    src/BMat.cpp
)

# include
target_include_directories(bdlearn
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# target
target_compile_features(bdlearn PRIVATE cxx_auto_type)
target_compile_options(bdlearn PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall>)

# test
add_subdirectory(test)
