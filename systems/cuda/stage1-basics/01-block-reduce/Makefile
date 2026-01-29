# Makefile for Block Reduce

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -arch=sm_70 -I..

# Target executable
TARGET = block_reduce

# Source files
SRCS = block_reduce.cu

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Run
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET)

.PHONY: all run clean
