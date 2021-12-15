#include <mkl.h>
#include <vector.h>
#include <stdio.h>
#include <stdlib.h>


Vector vector_new(double* data, int length){
    Vector newvector;
    newvector.data = data;
    newvector.length = length;
    return newvector;

}

Vector vector_alloc(){
    Vector newvector;
    newvector.data = NULL;
    newvector.length = 0;
    return newvector;
}

void set_data(Vector vec, double* data){
    if(vec.data != NULL){
        free(vec.data);
    }
    vec.data = data;
}

void copy(Vector src, Vector dst){
    *dst.data = *src.data;
    dst.length = src.length;
    return;
}

void set_length(Vector vec, int length){
    vec.length = length;
}

int get_length(Vector vec){
    return vec.length;
}

double *get_data(Vector vec){
    return vec.data;
}

void print_vec(Vector vec){
    for(int i = 0; i < vec.length; i++){
        printf("%f\n", vec.data[i]);
    }
}


void vector_free(Vector vec){
    if(vec.data != NULL){
        mkl_free((void*) vec.data);
        vec.data = NULL;
    }
    return;
}

void ptr_free(double* x){
    if(x != NULL){
        mkl_free(x);
        x = NULL;
    }
    return;
}


double* malloc_vector(size_t size, int i){
    return (double*) mkl_malloc(size, i);
}

void dealloc_vector(double* a){
    if(a != NULL)
        mkl_free((void*) a);
    return;
}

double *add_vector(Vector x, Vector y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdAdd(x.length, x.data, y.data, a);
    return a;
}

void iadd_vector(Vector x, Vector y){
    vdAdd(x.length, x.data, y.data, x.data);
}

void iadd_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, x.data);
}


double *add_scalar(Vector x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double),512);
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return a;
}

double* mul_vector(Vector x, Vector y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdMul(x.length, x.data, y.data, a);
    return a;
}

double* mul_scalar(Vector x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, scal, 0, 0, 1, a);
    return a;
}

void imul_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, scal, 0, 0, 1, x.data);
    return;
}
void imul_vector(Vector x, Vector y){
    vdMul(x.length, x.data, y.data, x.data);
    return;
}

Vector vec_from_ptr(int length, double* data){
    double* res = mkl_malloc(length * sizeof(double), 512);
    cblas_dcopy(length, data, 1, res, 1);
    Vector newvector;
    newvector.data = res;
    newvector.length = length;
    return newvector;
}

