CC = gcc
CPPC = g++
NVCC = nvcc

GPU = 0
CUDNN = 0
OPENCV = 0
OPENMP = 0
DEBUG = 0

ARCH = -gencode arch=compute_30,code=sm_30 \
       -gencode arch=compute_35,code=sm_35 \
       -gencode arch=compute_50,code=[sm_50,compute_50] \
       -gencode arch=compute_52,code=[sm_52,compute_52]

BUILD_DIR = build/
OBJ_DIR = $(BUILD_DIR)obj/
BIN_DIR = $(BUILD_DIR)bin/

INC_DIR = include/
SRC_DIR = src/
SCNN_DIR = $(SRC_DIR)scnn/

EXECUTABLE = $(BIN_DIR)serialcore

INCLUDES = -I$(INC_DIR)
CFLAGS = -O2 -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast
CPPFLAGS = $(CFLAGS)
LDFLAGS = -lm -pthread

ifeq ($(DEBUG),1)
    CFLAGS = -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -O0 -g -DDEBUG
    CPPFLAGS = $(CFLAGS)
endif

ifeq ($(OPENMP),1)
    CFLAGS += -fopenmp
    CPPFLAGS += -fopenmp
    LDFLAGS += -fopenmp
endif

CPP_SOURCES =
ifeq ($(OPENCV),1)
    CFLAGS += -DOPENCV
    CPPFLAGS += -DOPENCV
    INCLUDES += `pkg-config --cflags opencv`
    LDFLAGS += `pkg-config --libs opencv` -lstdc++
    CPP_SOURCES += $(wildcard $(CNN_DIR)*.cc)
endif

CU_SOURCES =
ifeq ($(GPU),1)
    CFLAGS += -DGPU
    CPPFLAGS += -DGPU
    INCLUDES += -I/usr/local/cuda/include/
    LDFLAGS += -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lstdc++
    CU_SOURCES += $(wildcard $(CNN_DIR)*.cu)
endif

ifeq ($(CUDNN),1)
    CFLAGS += -DCUDNN
    CPPFLAGS += -DCUDNN
    LDFLAGS += -lcudnn
endif

C_SOURCES = $(wildcard $(SRC_DIR)*.c $(SCNN_DIR)*.c)

OBJECTS = $(patsubst %.c,$(OBJ_DIR)%.o,$(C_SOURCES)) \
          $(patsubst %.cc,$(OBJ_DIR)%.o,$(CPP_SOURCES)) \
          $(patsubst %.cu,$(OBJ_DIR)%.o,$(CU_SOURCES))

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)%.o: %.cc
	@mkdir -p $(@D)
	$(CPPC) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(ARCH) $(INCLUDES) --compiler-options "$(CFLAGS)" -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -rf $(BUILD_DIR)

cpu:
	$(MAKE) GPU=0 CUDNN=0

gpu:
	$(MAKE) GPU=1 CUDNN=0

gpu-cudnn:
	$(MAKE) GPU=1 CUDNN=1

full:
	$(MAKE) GPU=1 CUDNN=1 OPENCV=1 OPENMP=1

dev:
	$(MAKE) DEBUG=1

.PHONY: all run clean print-sources cpu gpu gpu-cudnn full dev
