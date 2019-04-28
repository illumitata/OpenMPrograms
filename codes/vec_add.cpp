#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <ctime>
#include <cmath>

#include <omp.h>

#define VEC_SIZE 64000000
#define CACHE_LEN 64
typedef double my_type_t;

template <class T>
std::vector<T> vec_add_naive (std::vector<T> a, 
                              std::vector<T> b)
{
    std::vector<T> c(a.size());
    size_t i;
    size_t n = c.size();
    for (i = 0; i < n; i++)
        c[i] = a[i] + b[i];
    return c;
}

template <class T>
std::vector<T> vec_add_openmp (std::vector<T> a, 
                               std::vector<T> b)
{
    std::vector<T> c(a.size());  
    size_t n = c.size();
    #pragma omp parallel
    {
        size_t i;                           // private for each thread
        auto idx = omp_get_thread_num();    // manually get the index of running thread
        auto num = omp_get_num_threads();   // manually get the number of all threads
        for (i = idx; i < n; i += num)
            c[i] = a[i] + b[i];
    }
    return c;
}

template <class T>
std::vector<T> vec_add_openmp_for (std::vector<T> a,
                                   std::vector<T> b)
{
    std::vector<T> c(a.size());
    size_t i;   
    size_t n = c.size();
    // #pragma omp parallel
    // {
    //     #pragma omp for
    //     for (i = 0; i < n; i++)
    //         c[i] = a[i] + b[i];
    // }
    #pragma omp parallel for private(i)
        for (i = 0; i < n; i++)
            c[i] = a[i] + b[i];
    return c;
}

template <class T>
std::vector<T> vec_add_openmp_opt (std::vector<T> a, 
                                   std::vector<T> b)
{
    std::vector<T> c(a.size());
    size_t i;
    size_t n = c.size();
    size_t var_size = sizeof(T);
    size_t chunk = CACHE_LEN / var_size;
    #pragma omp parallel for private(i) schedule(guided, chunk) // static -> guided : for better performance
    for (i = 0; i < n; i++)
        c[i] = a[i] + b[i];
    return c;
}

template <class T>
bool ref_check (std::vector<T> a, std::vector<T> b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
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
    std::vector<my_type_t> c_ref = vec_add_naive<my_type_t>(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Reference:\t" << (end - start).count() << '\n';
    // Use basic OpenMP
    start = std::chrono::high_resolution_clock::now();
    std::vector<my_type_t> c_1 = vec_add_openmp<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Basic OpenMP:\t" << (end - start).count() << '\n';
    // Use opt-OpenMP
    start = std::chrono::high_resolution_clock::now();
    std::vector<my_type_t> c_3 = vec_add_openmp_opt<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Opt OpenMP:\t" << (end - start).count() << '\n';    
    // Use for-OpenMP
    start = std::chrono::high_resolution_clock::now();
    std::vector<my_type_t> c_2 = vec_add_openmp_for<my_type_t>(a, b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "For OpenMP:\t" << (end - start).count() << '\n';
    // Check againts the reference vector
    if (!(ref_check(c_ref, c_1)))
        std::cout << "Error: openmp1\n";
    if (!(ref_check(c_ref, c_2)))
        std::cout << "Error: openmp2\n";
    if (!(ref_check(c_ref, c_3)))
        std::cout << "Error: openmp3\n";
    return 0;
}