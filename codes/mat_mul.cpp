#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <ctime>
#include <cmath>

#include <omp.h>

#define MAT_SIZE 1200
#define CACHE_LEN 64
typedef long int my_type_t;

template <class T>
std::vector<std::vector<T>> mat_mul_naive (std::vector<std::vector<T>> a, 
                                           std::vector<std::vector<T>> b)
{
    std::vector<std::vector<T>> c(a.size(), std::vector<T>(a.size()));
    size_t i, j, k;
    size_t n = a.size();    
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];
    return c;
}

template <class T>
std::vector<std::vector<T>> mat_mul_trans (std::vector<std::vector<T>> a, 
                                           std::vector<std::vector<T>> b)
{
    std::vector<std::vector<T>> b_trans(a.size(), std::vector<T>(a.size()));
    std::vector<std::vector<T>> c(a.size(), std::vector<T>(a.size()));
    size_t i, j, k;
    size_t n = a.size();    
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            b_trans[i][j] = b[j][i];
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                c[i][j] += a[i][k] * b_trans[j][k];
    return c;
}

template <class T>
std::vector<std::vector<T>> mat_mul_flat (std::vector<std::vector<T>> a, 
                                         std::vector<std::vector<T>> b)
{
    size_t n = a.size();
    size_t i, j, k;
    std::vector<T> a_flat(n * n);
    std::vector<T> b_flat(n * n);
    std::vector<T> c_flat(n * n);
    std::vector<T> b_trans(n * n);
    // Flatten all input matrices
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a_flat[i * n + j] = a[i][j];
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            b_flat[i * n + j] = b[i][j];
    // Transpose B
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            b_trans[i * n + j] = b_flat[j * n + i];
    // Compute C = A * B
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                c_flat[i * n + j] += a_flat[i * n + k] * b_trans[j * n + k];
    // Unflatten just to show the impact
    std::vector<std::vector<T>> c(n, std::vector<T>(n));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            c[i][j] = c_flat[i * n + j];             
    return c;
}

template <class T>
std::vector<std::vector<T>> mat_mul_opt (std::vector<std::vector<T>> a, 
                                         std::vector<std::vector<T>> b)
{
    size_t n = a.size();
    size_t i, j, k;
    std::vector<T> a_flat(n * n);
    std::vector<T> b_flat(n * n);
    std::vector<T> c_flat(n * n);
    std::vector<T> b_trans(n * n);
    // Flatten all input matrices
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a_flat[i * n + j] = a[i][j];
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            b_flat[i * n + j] = b[i][j];
    // Transpose B in parallel
    size_t var_size = sizeof(T);
    size_t chunk = CACHE_LEN / var_size;
    #pragma omp parallel for shared(b_flat, b_trans) private(i, j) schedule(guided, chunk)
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            b_trans[i * n + j] = b_flat[j * n + i];
    //////////////////////////////////////
    // size_t num = omp_get_max_threads();
    // std::vector<T> b_tmp(num * num);
    // for (i = 0; i < n; i += num)
    // {
    //     for (j = 0; j < n; j += num)
    //     {
    //         size_t priv_i; // private for each thread
    //         #pragma omp parallel num_threads(num) private(priv_i) // shared(b_flat, b_tmp, b_trans)
    //         {
    //             auto idx = omp_get_thread_num(); // manually get the index of running thread
    //             for (priv_i = 0; priv_i < num; priv_i++)
    //                 b_tmp[priv_i * num + idx] = b_flat[i * n + idx * n + j + priv_i];
    //             #pragma omp barrier
    //             for (priv_i = 0; priv_i < num; priv_i++)
    //                 b_trans[j * n + idx * n + i + priv_i] = b_tmp[idx * num + priv_i];
    //         }
    //     }
    // }
    //////////////////////////////////////
    #pragma omp parallel for shared(a_flat, b_trans, c_flat) private(i, j, k) schedule(guided, chunk)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                c_flat[i * n + j] += a_flat[i * n + k] * b_trans[j * n + k];
            }
        }
    }
    // Unflatten just to show the impact
    std::vector<std::vector<T>> c(n, std::vector<T>(n));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            c[i][j] = c_flat[i * n + j];             
    return c;
}

template <class T>
bool ref_check (std::vector<std::vector<T>> a, 
                std::vector<std::vector<T>> b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
    {
        for (size_t j = 0; j < a.size(); j++)
        {
            if (a[i] != b[i])
                return false;
        }
    }
    return true;
}

int main()
{
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine eng(seed);
    std::uniform_int_distribution<> dist(-10, 10);
    // Generate random vectors
    std::vector<std::vector<my_type_t>> a(MAT_SIZE, std::vector<my_type_t>(MAT_SIZE));
    std::vector<std::vector<my_type_t>> b(MAT_SIZE, std::vector<my_type_t>(MAT_SIZE));
    for (size_t i = 0; i < MAT_SIZE; i++)
        std::generate(a[i].begin(), a[i].end(), [&](){ return (my_type_t)dist(eng); });
    for (size_t i = 0; i < MAT_SIZE; i++)
        std::generate(b[i].begin(), b[i].end(), [&](){ return (my_type_t)dist(eng); });
    // Make reference
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<my_type_t>> c_ref = mat_mul_naive<my_type_t>(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Reference:\t" << (end - start).count() << '\n';
    // // Make transpose
    // start = std::chrono::high_resolution_clock::now();
    // std::vector<std::vector<my_type_t>> c_1 = mat_mul_trans<my_type_t>(a, b);
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Transpose:\t" << (end - start).count() << '\n';
    // if (!(ref_check(c_ref, c_1)))
    //     std::cout << "Error: transpose\n";
    // Transpose flat version
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<my_type_t>> c_2 = mat_mul_flat<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Flat:\t\t" << (end - start).count() << '\n';
    if (!(ref_check(c_ref, c_2)))
        std::cout << "Error: flat\n";
    // Heavy lit version
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<my_type_t>> c_3 = mat_mul_opt<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "My opt:\t\t" << (end - start).count() << '\n';            
    if (!(ref_check(c_ref, c_3)))
        std::cout << "Error: my opt\n";
    return 0;
}