#include <stdio.h>
#include <unistd.h>

#include <mkl.h>
#include <omp.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>



sparse_matrix_t add(sparse_matrix_t m_a, sparse_matrix_t m_b);
void dot_add(sparse_matrix_t m, double* x, double* y);
