# Simplified Makefile.

APP=pt-hyperspectral
CXX=g++
SRC_MAIN=src
OUT_DIR=bin

#extra paths
LIB_TC=../libterraclear/src
LIB_TORCH1=/data/software/libtorch/include
LIB_TORCH2=/data/software/libtorch/include/torch/csrc/api/include

#include folders
INC=$(LIB_TC) $(LIB_TORCH1) $(LIB_TORCH2)
INC_PARAMS=$(foreach d, $(INC), -I$d)

#additional sources files
S1=$(LIB_TC)/filetools.cpp
S2=
SRC_EXT=$(S1) $(S2)

#lib paths
LP1=/data/software/libtorch/lib
LP2=
LIB_PATH=$(LP1) $(LP2)
LIB_PATH_PARAMS=$(foreach d, $(LIB_PATH), -L$d)

#link libs
LIBS=jsoncpp torch c10
LIBS_PARAMS=$(foreach d, $(LIBS), -l$d)

#compile
$(APP): $(SRC_MAIN)/$(APP).cpp 
	test -d bin || mkdir -p bin
	$(CXX) $(SRC_MAIN)/$(APP).cpp $(SRC_EXT) -o $(OUT_DIR)/$(APP) $(INC_PARAMS) $(LIB_PATH_PARAMS) $(LIBS_PARAMS)

clean:
	rm -rf bin
	
