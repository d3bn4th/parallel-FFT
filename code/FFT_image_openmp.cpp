#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <omp.h>

// OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

using namespace std;
using namespace cv;

typedef complex<float> base;

template <typename T>
ostream &operator<<(ostream &o, vector<T> v)
{
    if (v.size() > 0)
        o << v[0];
    for (unsigned i = 1; i < v.size(); i++)
        o << " " << v[i];
    return o << endl;
}

/**
 * Parallel FFT transform and inverse transform using OpenMP
 * Arguments: vector of complex numbers, invert flag
 * Performs inplace transform
 */
void fft(vector<base> &a, bool invert)
{
    int n = (int)a.size();
    
    // Bit reversal ordering
    for (int i = 1, j = 0; i < n; ++i)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1)
            j -= bit;
        j += bit;
        if (i < j)
            swap(a[i], a[j]);
    }

    // Iterative FFT with OpenMP parallelization
    for (int len = 2; len <= n; len <<= 1)
    {
        double ang = 2 * M_PI / len * (invert ? 1 : -1);
        base wlen(cos(ang), sin(ang));
        
        #pragma omp parallel for
        for (int i = 0; i < n; i += len)
        {
            base w(1);
            for (int j = 0; j < len / 2; ++j)
            {
                base u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
            a[i] /= n;
    }
}

/**
 * Performs 2D FFT using OpenMP
 * Takes vector of complex vectors, invert flag and verbose level
 * Performs inplace FFT transform on input vector
 */
void fft2D(vector<vector<base>> &a, bool invert, int verbose = 0)
{
    auto matrix = a;
    
    // Transform the rows
    if (verbose > 0)
        cout << "Transforming Rows" << endl;

    #pragma omp parallel for
    for (auto i = 0; i < matrix.size(); i++)
    {
        fft(matrix[i], invert);
    }

    // Prepare for transforming columns
    if (verbose > 0)
        cout << "Converting Rows to Columns" << endl;

    a = matrix;
    matrix.resize(a[0].size());
    for (int i = 0; i < matrix.size(); i++)
        matrix[i].resize(a.size());

    // Transpose matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            matrix[j][i] = a[i][j];
        }
    }
    
    if (verbose > 0)
        cout << "Transforming Columns" << endl;

    // Transform the columns
    #pragma omp parallel for
    for (auto i = 0; i < matrix.size(); i++)
        fft(matrix[i], invert);

    if (verbose > 0)
        cout << "Storing the result" << endl;

    // Store the result after transposing
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            a[j][i] = matrix[i][j];
        }
    }
}

/**
 * Function to multiply two polynomials using FFT
 * Takes two polynomials represented as vectors as input
 * Returns the product of two vectors
 */
vector<int> mult(vector<int> a, vector<int> b)
{
    // Create complex vectors from input vectors
    vector<base> fa(a.begin(), a.end()), fb(b.begin(), b.end());

    // Pad with zeros to make their size equal to power of 2
    size_t n = 1;
    while (n < max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;

    fa.resize(n), fb.resize(n);

    // Transform both a and b
    fft(fa, false), fft(fb, false);

    // Perform point-wise multiplication
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
        fa[i] *= fb[i];

    // Perform inverse transform
    fft(fa, true);

    // Save the real part as the result
    vector<int> res;
    res.resize(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
        res[i] = int(fa[i].real() + 0.5);

    return res;
}

/**
 * Function to perform JPEG compression on image using FFT
 * Takes image, threshold, and verbose level as input
 * Image is represented as vector<vector<uint8_t>>
 * Performs inplace compression on the input
 */
void compress_image(vector<vector<uint8_t>> &image, double threshold, int verbose = 1)
{
    // Convert image to complex type
    vector<vector<base>> complex_image(image.size(), vector<base>(image[0].size()));
    
    #pragma omp parallel for collapse(2)
    for (auto i = 0; i < image.size(); i++)
    {
        for (auto j = 0; j < image[0].size(); j++)
        {
            complex_image[i][j] = image[i][j];
        }
    }
    
    if (verbose == 1)
    {
        cout << "Input Image" << endl;
        cout << endl << endl;
    }
    if (verbose > 1)
    {
        cout << "Complex Image" << endl;
        cout << complex_image;
        cout << endl << endl;
    }

    // Perform 2D FFT on image
    fft2D(complex_image, false, verbose);

    if (verbose == 1)
    {
        cout << "Performing FFT on Image" << endl;
        cout << endl << endl;
    }

    // Threshold the FFT
    double maximum_value = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:maximum_value)
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            maximum_value = max(maximum_value, abs(complex_image[i][j]));
        }
    }
    
    threshold *= maximum_value;
    int count = 0;

    // Set values less than threshold to zero (compression step)
    #pragma omp parallel for collapse(2) reduction(+:count)
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            if (abs(complex_image[i][j]) < threshold)
            {
                count++;
                complex_image[i][j] = 0;
            }
        }
    }
    
    int zeros_count = 0;
    #pragma omp parallel for collapse(2) reduction(+:zeros_count)
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            if (abs(complex_image[i][j]) == 0)
            {
                zeros_count++;
            }
        }
    }
    
    cout << "Components removed (percent): " << ((zeros_count * 1.00 / (complex_image.size() * complex_image[0].size())) * 100) << endl;
    
    if (verbose > 1)
    {
        cout << "Thresholded Image" << endl;
        cout << endl << endl;
    }

    // Perform inverse FFT
    fft2D(complex_image, true, verbose);
    if (verbose > 1)
    {
        cout << "Inverted Image" << endl;
        cout << endl << endl;
    }
    
    // Convert to uint8 format (consider only real part)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            image[i][j] = uint8_t(complex_image[i][j].real() + 0.5);
        }
    }
    
    if (verbose > 0)
    {
        cout << "Compressed Image" << endl;
    }
}

int main()
{
    // Set number of OpenMP threads
    omp_set_num_threads(8); // Adjust based on your CPU cores
    
    Mat image_M;
    image_M = imread("squirrel.jpg", IMREAD_GRAYSCALE);
    if (!image_M.data)
    {
        cout << "Could not open or find the image" << endl;
        cout << "Please ensure 'squirrel.jpg' is in the current directory" << endl;
        return -1;
    }

    cv::imwrite("original.jpg", image_M);
    vector<vector<uint8_t>> image(image_M.rows, vector<uint8_t>(image_M.cols));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < image_M.rows; ++i)
        for (int j = 0; j < image_M.cols; ++j)
            image[i][j] = uint8_t(image_M.at<uint8_t>(i, j));

    freopen("out.txt", "w", stdout);
    for(double thresh = 0.000001; thresh < 1; thresh *= 10)
    {
        cout << "For thresh= " << thresh << endl;
        
        // Create a copy of the original image for each threshold
        vector<vector<uint8_t>> image_copy = image;
        compress_image(image_copy, thresh, 0);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < image_M.rows; ++i)
            for (int j = 0; j < image_M.cols; ++j)
                image_M.at<uint8_t>(i, j) = image_copy[i][j];
                
        string s = "compressed_";
        s = s + to_string(thresh);
        s += ".jpg";
        cv::imwrite(s, image_M);
    }

    return 0;
}
