CC=g++
HALIDE_PATH=/home/dhy1996/Halide
FLAGS=-std=c++11 -g -fno-omit-frame-pointer -I $(HALIDE_PATH)/tools -I/usr/include/libpng16 -I/usr/include/libpng16/.. -O0
FLAGS_SUF=$(HALIDE_PATH)/bin/build/GenGen.o -I$(HALIDE_PATH)/include -Lbin -lHalide -ldl -lpthread -lz  -rdynamic -Wl,--rpath=$(HALIDE_PATH)/bin -lpng16  -ljpeg
BIN_DIR=../../bin

all: batchim2col batchcol2imaccum batchmatmulabr
.PHONY: all

batchim2col: BatchIm2Col.cpp
	$(CC) $(FLAGS) BatchIm2Col.cpp $(FLAGS_SUF) -o $(BIN_DIR)/batchim2col

batchcol2imaccum: BatchCol2ImAccum.cpp
	$(CC) $(FLAGS) BatchCol2ImAccum.cpp $(FLAGS_SUF) -o $(BIN_DIR)/batchcol2imaccum

batchmatmulabr: BatchMatMul_ABr.cpp
	$(CC) $(FLAGS) BatchMatMul_ABr.cpp $(FLAGS_SUF) -o $(BIN_DIR)/batchmatmulabr