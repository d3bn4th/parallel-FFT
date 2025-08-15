# OpenMP-based Parallel FFT Implementation

This is a macOS-compatible version of the parallel FFT project using OpenMP instead of CUDA. OpenMP provides excellent cross-platform parallel processing capabilities and works great on macOS.

## Features

- **Cross-platform compatibility**: Works on macOS, Linux, and Windows
- **CPU-based parallelism**: Utilizes all CPU cores efficiently
- **Easy compilation**: No GPU dependencies required
- **Same functionality**: Image compression and polynomial multiplication

## Prerequisites

### 1. Install Xcode Command Line Tools
```bash
xcode-select --install
```

### 2. Install OpenCV
```bash
# Using Homebrew (recommended)
brew install opencv

# Or using conda
conda install opencv
```

### 3. Verify OpenMP Support
OpenMP should be included with Xcode Command Line Tools on macOS.

## Quick Start

### 1. Build the Programs
```bash
cd code
make all
```

### 2. Run Polynomial Multiplication
```bash
./fft_mult_openmp
```

### 3. Run Image Compression
```bash
# Copy a test image first
cp ../images/squirrel.jpg .
./fft_image_openmp
```

## Manual Compilation

If you prefer to compile manually:

### Polynomial Multiplication
```bash
g++ -std=c++11 -O3 -fopenmp -o fft_mult_openmp FFT_multiplication_openmp.cpp $(pkg-config --cflags --libs opencv4)
```

### Image Compression
```bash
g++ -std=c++11 -O3 -fopenmp -o fft_image_openmp FFT_image_openmp.cpp $(pkg-config --cflags --libs opencv4)
```

## Key Differences from CUDA Version

### 1. Parallelization Strategy
- **CUDA**: GPU-based parallelism with thousands of threads
- **OpenMP**: CPU-based parallelism with `#pragma omp` directives

### 2. Memory Management
- **CUDA**: Explicit GPU memory allocation and transfer
- **OpenMP**: Standard C++ memory management

### 3. Thread Control
- **CUDA**: Complex thread/block organization
- **OpenMP**: Simple thread count setting with `omp_set_num_threads()`

## Performance Considerations

### Thread Count Optimization
Adjust the number of threads based on your CPU:
```cpp
omp_set_num_threads(8); // For 8-core CPU
```

### Compiler Optimizations
The Makefile includes `-O3` for maximum optimization. You can experiment with:
- `-O2` for balanced optimization
- `-march=native` for CPU-specific optimizations

## Available Makefile Targets

```bash
make all              # Build all programs
make fft_image_openmp # Build image compression only
make fft_mult_openmp  # Build polynomial multiplication only
make test             # Run basic tests
make clean            # Remove build files
make help             # Show all options
```

## Troubleshooting

### 1. OpenCV Not Found
```bash
# Install OpenCV
brew install opencv

# Or specify OpenCV path manually
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### 2. OpenMP Not Supported
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

### 3. Compilation Errors
```bash
# Clean and rebuild
make clean
make all
```

## Performance Comparison

The OpenMP version provides:
- **Easier setup**: No GPU drivers or CUDA installation required
- **Cross-platform**: Works on any system with a C++ compiler
- **Good performance**: Efficient CPU utilization
- **Scalability**: Scales with CPU cores

While CUDA may offer better performance for very large datasets on high-end GPUs, OpenMP provides excellent performance for most use cases and is much easier to deploy.

## Example Output

### Polynomial Multiplication
```
A = 1 1
B = 1 2 3
A * B = 1 3 5 3

=== Performance Comparison ===
Generating random polynomials of size 1000
For threads= 1
1 1234 5678 1
For threads= 2
2 678 3456 1
For threads= 4
4 456 2345 1
For threads= 8
8 345 1987 1
```

### Image Compression
```
For thresh= 1e-06
Components removed (percent): 12.34
For thresh= 1e-05
Components removed (percent): 23.45
...
```

## Files

- `FFT_image_openmp.cpp` - Image compression using OpenMP
- `FFT_multiplication_openmp.cpp` - Polynomial multiplication using OpenMP
- `Makefile` - Build automation
- `README_OpenMP.md` - This file

## Original CUDA Files (for reference)

- `FFT_image.cu` - Original CUDA image compression
- `FFT_multiplication.cu` - Original CUDA polynomial multiplication
