#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <ctime>
#include <cmath>

#include <omp.h>

#define VEC_SIZE 128000000
#define CACHE_LEN 64
typedef long int my_type_t;

template <class T>
T vec_dot_naive (std::vector<T> a, std::vector<T> b)
{
    T res = 0;
    size_t i;
    size_t n = a.size();    
    for (i = 0; i < n; i++)
        res += a[i] * b[i];
    return res;
}

// template <class T>
// T vec_dot_openmp_ug (std::vector<T> a, std::vector<T> b)
// {
//     T res;
//     size_t idx;
//     size_t n = a.size();
//     for (h = 1; h <= (int)log2(n); h++)
//     {
//     }
//     return res;
// }

template <class T>
T vec_dot_openmp_my_one (std::vector<T> a, std::vector<T> b)
{
    T res = 0;
    size_t n = a.size();
    #pragma omp parallel
    {
        size_t i;
        T local_res;
        #pragma omp for
        for (i = 0; i < n; i++)
        {
            local_res = 0;
            local_res += a[i] * b[i];
            #pragma omp critical
            res += local_res;
        }
    }
    return res;
}

template <class T>
T vec_dot_openmp_my_two (std::vector<T> a, std::vector<T> b)
{
    T res = 0;
    T local_res;
    size_t n = a.size();
    size_t var_size = sizeof(T);
    size_t chunk = CACHE_LEN / var_size;
    #pragma omp parallel private(local_res)
    {
        size_t i;
        local_res = 0;
        #pragma omp for schedule(guided, chunk)
        for (i = 0; i < n; i++)
            local_res += a[i] * b[i];
        #pragma omp atomic
        res += local_res;
    }
    return res;
}

template <class T>
T vec_dot_openmp_opt (std::vector<T> a, std::vector<T> b)
{
    T res = 0;
    size_t i;
    size_t n = a.size();
    #pragma omp parallel for shared(a, b) private(i) reduction(+:res)
    for (i = 0; i < n; i++)
        res += a[i] * b[i];
    return res;
}

template <class T>
bool ref_check (T a, T b)
{
    return a == b ? true : false;
}

int main()
{
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine eng(seed);
    std::uniform_int_distribution<> dist(-10, 10);
    // Generate random vectors
    std::vector<my_type_t> a(VEC_SIZE);
    std::vector<my_type_t> b(VEC_SIZE);
    std::generate(a.begin(), a.end(), [&](){ return (my_type_t)dist(eng); });
    std::generate(b.begin(), b.end(), [&](){ return (my_type_t)dist(eng); });
    // Make reference
    auto start = std::chrono::high_resolution_clock::now();
    my_type_t c_ref = vec_dot_naive<my_type_t>(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Reference:\t" << (end - start).count() << '\n';
    // Use my one OpenMP
    start = std::chrono::high_resolution_clock::now();
    my_type_t c_1 = vec_dot_openmp_my_one<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "My one OpenMP:\t" << (end - start).count() << '\n'; 
    // Use my two OpenMP
    start = std::chrono::high_resolution_clock::now();
    my_type_t c_2 = vec_dot_openmp_my_two<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "My two OpenMP:\t" << (end - start).count() << '\n'; 
    // Use opt OpenMP
    start = std::chrono::high_resolution_clock::now();
    my_type_t c_3 = vec_dot_openmp_opt<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Opt OpenMP:\t" << (end - start).count() << '\n';       
    // Check againts the reference vector
    if (!(ref_check(c_ref, c_1)))
        std::cout << "Error: openmp1\n";
    if (!(ref_check(c_ref, c_2)))
        std::cout << "Error: openmp2\n";
    if (!(ref_check(c_ref, c_3)))
        std::cout << "Error: openmp3\n";
    return 0;
}