# Makefile for OpenMP-based FFT implementation
# Compatible with macOS

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

# Default target
all: fft_image_openmp fft_mult_openmp

# OpenMP Image Compression
fft_image_openmp: FFT_image_openmp.cpp
	$(CXX) $(CXXFLAGS) -o fft_image_openmp FFT_image_openmp.cpp $(OPENCV_FLAGS)

# OpenMP Polynomial Multiplication
fft_mult_openmp: FFT_multiplication_openmp.cpp
	$(CXX) $(CXXFLAGS) -o fft_mult_openmp FFT_multiplication_openmp.cpp $(OPENCV_FLAGS)

# Original CUDA versions (for reference)
fft_image_cuda: FFT_image.cu
	nvcc -o fft_image_cuda FFT_image.cu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

fft_mult_cuda: FFT_multiplication.cu
	nvcc -o fft_mult_cuda FFT_multiplication.cu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

# Clean build files
clean:
	rm -f fft_image_openmp fft_mult_openmp fft_image_cuda fft_mult_cuda *.o

# Install dependencies (macOS)
install-deps:
	# Install OpenCV using Homebrew
	brew install opencv
	# Install OpenMP (usually comes with Xcode Command Line Tools)
	xcode-select --install

# Run tests
test: fft_image_openmp fft_mult_openmp
	@echo "Testing polynomial multiplication..."
	./fft_mult_openmp
	@echo "Testing image compression (requires squirrel.jpg)..."
	@if [ -f squirrel.jpg ]; then \
		./fft_image_openmp; \
	else \
		echo "Warning: squirrel.jpg not found. Copy an image to test compression."; \
	fi

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all OpenMP programs"
	@echo "  fft_image_openmp - Build image compression program"
	@echo "  fft_mult_openmp  - Build polynomial multiplication program"
	@echo "  install-deps     - Install dependencies (macOS)"
	@echo "  test             - Run basic tests"
	@echo "  clean            - Remove build files"
	@echo "  help             - Show this help"

.PHONY: all clean install-deps test help
