CC=g++
HALIDE_PATH=/home/dhy1996/Halide
FLAGS=-std=c++11 -g -fno-omit-frame-pointer -I $(HALIDE_PATH)/tools -I/usr/include/libpng16 -I/usr/include/libpng16/.. -O0
FLAGS_SUF=$(HALIDE_PATH)/bin/build/GenGen.o -I$(HALIDE_PATH)/include -Lbin -lHalide -ldl -lpthread -lz  -rdynamic -Wl,--rpath=$(HALIDE_PATH)/bin -lpng16  -ljpeg
BIN_DIR=../../bin

batchim2col:
	$(CC) $(FLAGS) BatchIm2Col.cpp $(FLAGS_SUF) -o $(BIN_DIR)/batchim2col