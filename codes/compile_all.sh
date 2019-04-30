#!/bin/bash
g++ vec_add.cpp -o vec_add -fopenmp # -O3 -funroll-loops
g++ vec_dot.cpp -o vec_dot -fopenmp # -O3 -funroll-loops
g++ mat_mul.cpp -o mat_mul -fopenmp # -O3 -funroll-loops