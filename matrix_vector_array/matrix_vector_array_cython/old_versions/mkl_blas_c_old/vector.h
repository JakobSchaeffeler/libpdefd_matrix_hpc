#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <sched.h>

#include <mkl.h>
#include <omp.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct _Vector{
    int length;
    double* data;
} Vector;

Vector vector_new( double* data, int length);
Vector vector_alloc(void);
void set_length(Vector vec, int length);
void set_data(Vector vec, double* data);
double *get_data(Vector vec);
void print_vec(Vector vec);
int get_length(Vector vec);
void vector_free(Vector vec);
void ptr_free(double* x);
double* malloc_vector(size_t size);
double* add_vector(Vector x, Vector y);
double* add_scalar(Vector x, double scal);
void iadd_scalar(Vector x, double scal);
void iadd_vector(Vector x, Vector y);

double* mul_vector(Vector x, Vector y);
double* mul_scalar(Vector x, double scal);
void imul_scalar(Vector x, double scal);
void imul_vector(Vector x, Vector y);

Vector copy(Vector x);
Vector vec_from_ptr(int length, double*    data);
