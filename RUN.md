# 1. Install dependencies
brew install opencv
xcode-select --install

# 2. Build the programs
cd code
make all

# 3. Run polynomial multiplication
./fft_mult_openmp

# 4. Run image compression (copy test image first)
cp ../images/squirrel.jpg .
./fft_image_openmp