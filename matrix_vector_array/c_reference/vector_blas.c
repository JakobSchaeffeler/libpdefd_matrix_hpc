#include <mkl.h>
#include "vector_blas.h"
#include <omp.h>
#include <stdio.h>

Vector_blas vector_new_blas(double* data, int length){
    Vector_blas newvector;
    newvector.data = data;
    newvector.length = length;
    return newvector;

}

Vector_blas vector_alloc_blas(){
    Vector_blas newvector;
    newvector.data = NULL;
    newvector.length = 0;
    return newvector;
}

void set_data_blas(Vector_blas vec, double* data){
    if(vec.data != NULL){
        mkl_free(vec.data);
    }
    vec.data = data;
}

void set_length_blas(Vector_blas vec, int length){
    vec.length = length;
}

int get_length_blas(Vector_blas vec){
    return vec.length;
}

double *get_data_blas(Vector_blas vec){
    return vec.data;
}

void print_vec_blas(Vector_blas vec){
    for(int i = 0; i < vec.length; i++){
        printf("%f\n", vec.data[i]);
    }
}

double* malloc_vector_blas(int size, int i){
    return (double*) mkl_malloc(size, i);
}

void vector_free_blas(Vector_blas vec){
    if(vec.data != NULL){
        mkl_free(vec.data);
    }
    return;
}


double* add_vector_blas(Vector_blas x, Vector_blas y){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, y.data, 1, a, 1);
    cblas_daxpy(x.length, 1.0, x.data, 1, a, 1);
    return a;
}

double *add_scalar_blas(Vector_blas x, double scal){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return a;
}

void iadd_vector_blas(Vector_blas x, Vector_blas y){
	cblas_daxpy(x.length, 1.0, y.data, 1, x.data, 1);
}

void iadd_scalar_blas(Vector_blas x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, x.data);
}

void iadd_vector_blas_blocked(Vector_blas x, Vector_blas y){
    if(x.length > 1000000){
        int n = (int) x.length / 500000;
        #pragma omp parallel for
        for(int i = 0; i < n; i++){
            cblas_daxpy(500000, 1.0, &y.data[i*500000], 1, &x.data[i*500000], 1);
        }
        cblas_daxpy(x.length - 500000*n, 1.0, &y.data[n*500000], 1, &x.data[n*500000], 1);

    }
}

double* mul_vector_blas(Vector_blas x, Vector_blas y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    cblas_dsbmv(101, 122, x.length, 0, 1.0, x.data, 1, y.data, 1,
    0.0, a, 1);
    return a;
}
double* mul_scalar_blas(Vector_blas x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    cblas_daxpby(x.length, scal, x.data, 1, 0.0, a, 1);
    return a;
}

void imul_scalar_blas(Vector_blas x, double scal){
    cblas_dscal(x.length, scal, x.data, 1);
    return;
}
void imul_vector_blas(Vector_blas x, Vector_blas y){
    cblas_dsbmv(101, 122, x.length, 0, 1.0, x.data, 1, y.data, 1, 0.0, x.data, 1);
    return;
}

double* div_vector_blas(Vector_blas x, Vector_blas y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, x.data,y.data, a);
    return a;
}
double* div_scalar_blas(Vector_blas x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    cblas_daxpby(x.length, 1/scal, x.data, 1, 0.0, a, 1);
    return a;
}

void idiv_vector_blas(Vector_blas x, Vector_blas y){
    vdDiv(x.length, x.data,y.data, x.data);
    return;
}
void idiv_scalar_blas(Vector_blas x, double scal){
    cblas_daxpby(x.length, 1/scal, x.data, 1, 0.0, x.data, 1);
    return;
}

double* rdiv_vector_blas(Vector_blas x, Vector_blas y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, y.data, x.data, a);
    return a;
}
double* rdiv_scalar_blas(Vector_blas x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdInv(x.length, x.data, a);
    cblas_dscal(x.length, scal, a, 1);
    return a;
}



Vector_blas copy_blas(Vector_blas x){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, x.data, 1, a, 1);
    return vector_new_blas(a,x.length);
}

