#include "vector.h"


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

Vector add_vector(Vector x, Vector y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdAdd(x.length, x.data, y.data, a);
    return vector_new(a, x.length);
}

void iadd_vector(Vector x, Vector y){
    vdAdd(x.length, x.data, y.data, x.data);
}

void iadd_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, x.data);
}


Vector add_scalar(Vector x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double),512);
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return vector_new(a, x.length);
}

Vector mul_vector(Vector x, Vector y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdMul(x.length, x.data, y.data, a);
    return vector_new(a, x.length);
}

Vector mul_scalar(Vector x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, scal, 0, 0, 1, a);
    return vector_new(a, x.length);
}

void imul_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, scal, 0, 0, 1, x.data);
    return;
}
void imul_vector(Vector x, Vector y){
    vdMul(x.length, x.data, y.data, x.data);
    return;
}

Vector sub_vector(Vector x, Vector y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdSub(x.length, x.data, y.data, a);
    return vector_new(a, x.length);
}

Vector sub_scalar(Vector x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 1, -scal, 0, 1, a);
    return vector_new(a, x.length);
}

void isub_vector(Vector x, Vector y){
    vdSub(x.length, x.data, y.data, x.data);
}

void isub_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, -scal, 0, 1, x.data);
}

Vector rsub_vector(Vector x, Vector y){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdSub(x.length, y.data, x.data, a);
    return vector_new(a, x.length);
}

Vector rsub_scalar(Vector x, double scal){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, -1, scal, 0, 1, a);
    return vector_new(a, x.length);
}

Vector pow_vector(Vector x, Vector y){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdPow(x.length, x.data, y.data, a);
    return vector_new(a, x.length);
}

Vector pow_scalar(Vector x, double scal){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdPowx(x.length, x.data, scal, a);
    return vector_new(a, x.length);
}

Vector truediv_vector(Vector x, Vector y){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, x.data, y.data, a);
    return vector_new(a, x.length);
}

Vector truediv_scalar(Vector x, double scal){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 1/scal, 0, 0, 1, a);
    return vector_new(a, x.length);
}

Vector rtruediv_vector(Vector x, Vector y){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, y.data, x.data, a);
    return vector_new(a, x.length);
}

Vector rtruediv_scalar(Vector x, double scal){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 0, scal, 1, 0, a);
    return vector_new(a, x.length);
}

void itruediv_vector(Vector x, Vector y){
    vdDiv(x.length, x.data, y.data, x.data);
    return;
}

void itruediv_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1/scal, 0, 0, 1, x.data);
    return;
}

Vector neg(Vector x){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, -1, 0, 0, 1, a);
    return vector_new(a, x.length);
}

Vector copy(Vector x){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, x.data, 1, a, 1);
    return vector_new(a, x.length);
}

Vector vec_from_ptr(int length, double* data){
    double* res = mkl_malloc(length * sizeof(double), 512);
    cblas_dcopy(length, data, 1, res, 1);
    Vector newvector;
    newvector.data = res;
    newvector.length = length;
    return newvector;
}
