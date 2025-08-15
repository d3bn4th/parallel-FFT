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
 * Sequential FFT class for comparison
 */
class FFT
{
public:
    /**
     * Sequential FFT transform and inverse transform
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

        // Iterative FFT
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = 2 * M_PI / len * (invert ? 1 : -1);
            base wlen(cos(ang), sin(ang));
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
            for (int i = 0; i < n; ++i)
                a[i] /= n;
    }

    /**
     * Performs 2D FFT sequentially
     */
    void fft2D(vector<vector<base>> &a, bool invert, int verbose = 0)
    {
        auto matrix = a;
        
        // Transform the rows
        if (verbose > 0)
            cout << "Transforming Rows" << endl;

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
        for (auto i = 0; i < matrix.size(); i++)
            fft(matrix[i], invert);

        if (verbose > 0)
            cout << "Storing the result" << endl;

        // Store the result after transposing
        for (int i = 0; i < a.size(); i++)
        {
            for (int j = 0; j < a[0].size(); j++)
            {
                a[j][i] = matrix[i][j];
            }
        }
    }

    /**
     * Function to multiply two polynomials using sequential FFT
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
        for (size_t i = 0; i < n; ++i)
            fa[i] *= fb[i];

        // Perform inverse transform
        fft(fa, true);

        // Save the real part as the result
        vector<int> res;
        res.resize(n);
        for (size_t i = 0; i < n; ++i)
            res[i] = int(fa[i].real() + 0.5);

        return res;
    }
};

#define N 1000

int main()
{
    // Set number of OpenMP threads
    omp_set_num_threads(8); // Adjust based on your CPU cores
    
    // Simple polynomial multiplication example
    vector<int> a = {1, 1};
    vector<int> b = {1, 2, 3};
    auto multiplier = FFT();
    cout << "A = " << a;
    cout << "B = " << b;
    cout << "A * B = " << multiplier.mult(a, b) << endl;
    
    // Performance comparison between parallel and sequential
    cout << "\n=== Performance Comparison ===" << endl;
    cout << "Generating random polynomials of size " << N << endl;
    
    std::vector<int> fa(N);
    std::generate(fa.begin(), fa.end(), std::rand);
    std::vector<int> fb(N);
    std::generate(fb.begin(), fb.end(), std::rand);
    
    freopen("out.txt", "w", stdout);
    
    // Test with different thread counts
    for(int threads = 1; threads <= 16; threads *= 2)
    {
        omp_set_num_threads(threads);
        cerr << "For threads= " << threads << endl;
        
        // Parallel version
        auto start = high_resolution_clock::now(); 
        auto result_parallel = mult(fa, fb);
        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start); 
        cout << threads << " " << duration.count();

        // Sequential version
        auto multiplier_seq = FFT();
        start = high_resolution_clock::now(); 
        auto result_sequential = multiplier_seq.mult(fa, fb);
        stop = high_resolution_clock::now(); 
        duration = duration_cast<microseconds>(stop - start); 
        cout << " " << duration.count();

        // Verify results match
        bool results_match = (result_parallel == result_sequential);
        cout << " " << results_match << endl;
        
        if (!results_match) {
            cout << "WARNING: Results don't match!" << endl;
        }
    }
    
    return 0;
}
