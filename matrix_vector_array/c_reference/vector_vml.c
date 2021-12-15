#include <mkl.h>
#include "vector_vml.h"
#include <stdio.h>
#include <stdlib.h>


Vector_vml vector_new_vml(double* data, int length){
    Vector_vml newvector;
    newvector.data = data;
    newvector.length = length;
    return newvector;

}

Vector_vml vector_alloc_vml(){
    Vector_vml newvector;
    newvector.data = NULL;
    newvector.length = 0;
    return newvector;
}

void set_data_vml(Vector_vml vec, double* data){
    if(vec.data != NULL){
        free(vec.data);
    }
    vec.data = data;
}

void copy_vml(Vector_vml src, Vector_vml dst){
    *dst.data = *src.data;
    dst.length = src.length;
    return;
}

void set_length_vml(Vector_vml vec, int length){
    vec.length = length;
}

int get_length_vml(Vector_vml vec){
    return vec.length;
}

double *get_data_vml(Vector_vml vec){
    return vec.data;
}

void print_vec_vml(Vector_vml vec){
    for(int i = 0; i < vec.length; i++){
        printf("%f\n", vec.data[i]);
    }
}


void vector_free_vml(Vector_vml vec){
    if(vec.data != NULL){
        mkl_free((void*) vec.data);
        vec.data = NULL;
    }
    return;
}

double* malloc_vector_vml(int size, int i){
    return (double*) mkl_malloc(size, i);
}

void dealloc_vector_vml(double* a){
    if(a != NULL)
        mkl_free((void*) a);
    return;
}

double *add_vector_vml(Vector_vml x, Vector_vml y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdAdd(x.length, x.data, y.data, a);
    return a;
}

void iadd_vector_vml(Vector_vml x, Vector_vml y){
    vdAdd(x.length, x.data, y.data, x.data);
}

void iadd_scalar_vml(Vector_vml x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, x.data);
}


double *add_scalar_vml(Vector_vml x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double),512);
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return a;
}

double* mul_vector_vml(Vector_vml x, Vector_vml y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdMul(x.length, x.data, y.data, a);
    return a;
}

double* mul_scalar_vml(Vector_vml x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, scal, 0, 0, 1, a);
    return a;
}

void imul_scalar_vml(Vector_vml x, double scal){
    vdLinearFrac(x.length, x.data, x.data, scal, 0, 0, 1, x.data);
    return;
}
void imul_vector_vml(Vector_vml x, Vector_vml y){
    vdMul(x.length, x.data, y.data, x.data);
    return;
}
double* div_vector_vml(Vector_vml x, Vector_vml y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, x.data, y.data, a);
    return a;
}

double* div_scalar_vml(Vector_vml x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 1/scal, 0, 0, 1, a);
    return a;
}

void idiv_scalar_vml(Vector_vml x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1/scal, 0, 0, 1, x.data);
    return;
}
void idiv_vector_vml(Vector_vml x, Vector_vml y){
    vdDiv(x.length, x.data, y.data, x.data);
    return;
}
double* rdiv_vector_vml(Vector_vml x, Vector_vml y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, y.data, x.data, a);
    return a;
}

double* rdiv_scalar_vml(Vector_vml x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 0, scal, 1, 0, a);
    return a;
}


