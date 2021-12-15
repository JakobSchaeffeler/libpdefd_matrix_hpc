#include <mkl.h>
typedef struct _Vector_blas{
    int length;
    double* data;
} Vector_blas;

Vector_blas vector_new_blas( double* data, int length);
Vector_blas vector_alloc_blas(void);
void set_length_blas(Vector_blas vec, int length);
void set_data_blas(Vector_blas vec, double* data);
double *get_data_blas(Vector_blas vec);
void print_vec_blas(Vector_blas vec);
int get_length_blas(Vector_blas vec);
void vector_free_blas(Vector_blas vec);
double* malloc_vector_blas(int size, int i);
double* add_vector_blas(Vector_blas x, Vector_blas y);
double* add_scalar_blas(Vector_blas x, double scal);
void iadd_scalar_blas(Vector_blas x, double scal);
void iadd_vector_blas(Vector_blas x, Vector_blas y);
void iadd_vector_blas_blocked(Vector_blas x, Vector_blas y);


double* mul_vector_blas(Vector_blas x, Vector_blas y);
double* mul_scalar_blas(Vector_blas x, double scal);
void imul_scalar_blas(Vector_blas x, double scal);
void imul_vector_blas(Vector_blas x, Vector_blas y);

double* div_vector_blas(Vector_blas x, Vector_blas y);
double* div_scalar_blas(Vector_blas x, double scal);
double* rdiv_vector_blas(Vector_blas x, Vector_blas y);
double* rdiv_scalar_blas(Vector_blas x, double scal);

void idiv_scalar_blas(Vector_blas x, double scal);
void idiv_vector_blas(Vector_blas x, Vector_blas y);


Vector_blas copy(Vector_blas x);
