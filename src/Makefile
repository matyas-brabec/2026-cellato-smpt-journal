CXX = g++
CXX_STD = c++20
NVCC = nvcc
CXXFLAGS = -std=$(CXX_STD) -DNDEBUG -Wall -Wextra -Wpedantic -O3 -march=native -I../include
NVCC_CC_BIN = $(shell which $(CXX))
NVCCFLAGS = -std=$(CXX_STD) -DNDEBUG -Wreorder -Wext-lambda-captures-this -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -I../include -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -ccbin $(NVCC_CC_BIN) -Xcompiler -std=$(CXX_STD),-Wall,-Wextra,-march=native

CUDA_HOME = /usr/local/cuda

INCLUDE_DIRS = -I../include -I$(CUDA_HOME)/include
LIB_DIRS = -L$(CUDA_HOME)/lib64
LIBS = -lcudart

DEFINES =
ifeq ($(BUILD_TYPE), BENCHMARK)
    DEFINES += -DBENCHMARK_COMPILE
else ifeq ($(BUILD_TYPE), VERIFICATION)
    DEFINES += -DVERIFICATION_COMPILE
endif

# Directories
SRC_DIR = .
OBJ_DIR = ../bin
BIN_DIR = ../bin
CUDA_DIRS = ../include/evaluators ../include/traversers/cuda ./game_of_life ./excitable ./wire ./fire ./brian ./maze ./fluid ./critters ./traffic ./cyclic

# Source and object files
MAIN_SRC = $(SRC_DIR)/main.cpp
MAIN_OBJ = $(OBJ_DIR)/main.o
TEST_SRC = $(SRC_DIR)/test_runner.cpp
TEST_OBJ = $(OBJ_DIR)/test_runner.o

# CUDA source and object files
CUDA_SRCS = $(foreach dir,$(CUDA_DIRS),$(wildcard $(dir)/*.cu))
CUDA_OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(notdir $(CUDA_SRCS)))

# Main targets
TARGET = $(BIN_DIR)/cellato
TEST_TARGET = $(BIN_DIR)/cellato_tests

# Rules
all: directories $(TARGET) $(TEST_TARGET)

benchmark:
	@$(MAKE) -j$(shell nproc) all BUILD_TYPE=BENCHMARK

verify:
	@$(MAKE) -j$(shell nproc) all BUILD_TYPE=VERIFICATION

COMPILE_TO_BENCHMARK: benchmark
COMPILE_TO_VERIFY: verify


directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Add directories dependency to object file creation rules
$(MAIN_OBJ): $(MAIN_SRC) | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE_DIRS) $(DEFINES)

$(TEST_OBJ): $(TEST_SRC) | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE_DIRS) $(DEFINES)

# Pattern rule for CUDA files with vpath to find them
vpath %.cu $(CUDA_DIRS)
$(OBJ_DIR)/%.o: %.cu | directories
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(DEFINES)

# Create separate executables for main app and tests
$(TARGET): $(MAIN_OBJ) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB_DIRS) $(LIBS)

$(TEST_TARGET): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB_DIRS) $(LIBS)

# Debug rule to print variables
print-%:
	@echo $* = $($*)

run: $(TARGET)
	$(TARGET) $(ARGS)

unit_test: $(TEST_TARGET)
	$(TEST_TARGET) $(ARGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all run test clean directories print-%
