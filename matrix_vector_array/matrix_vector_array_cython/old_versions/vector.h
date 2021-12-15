#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <sched.h>

#include <mkl.h>
#include <omp.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <numa.h>


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
void ptr_mkl_free(double* x);

double* malloc_vector(int size);
double* mkl_malloc_vector(int size);
double* mkl_realloc_vector(double* a, int size);
double* numa_alloc_vector(int size);
void numa_free_vector(double* a,  int size);


Vector add_vector(Vector x, Vector y);
Vector add_scalar(Vector x, double scal);

void iadd_scalar(Vector x, double scal);
void iadd_vector(Vector x, Vector y);
void iadd_double_vector(double* x, double* y, int n);

Vector mul_vector(Vector x, Vector y);
Vector mul_scalar(Vector x, double scal);

void imul_scalar(Vector x, double scal);
void imul_double_scalar(double* x, double scal, int n);
void imul_vector(Vector x, Vector y);

Vector sub_vector(Vector x, Vector y);
Vector sub_scalar(Vector x, double scal);

void isub_vector(Vector x, Vector y);
void isub_scalar(Vector x, double scal);

Vector rsub_vector(Vector x, Vector y);
Vector rsub_scalar(Vector x, double scal);

Vector pow_vector(Vector x, Vector y);
Vector pow_scalar(Vector x, double scal);

Vector truediv_vector(Vector x, Vector y);
Vector truediv_scalar(Vector x, double scal);

Vector rtruediv_vector(Vector x, Vector y);
Vector rtruediv_scalar(Vector x, double scal);

void itruediv_vector(Vector x, Vector y);
void itruediv_scalar(Vector x, double scal);

Vector neg(Vector x);

Vector copy(Vector x);
void copy_double(double* x, double* y, int n);
Vector vec_from_ptr(int length, double* data);

void set_threads(int i);
