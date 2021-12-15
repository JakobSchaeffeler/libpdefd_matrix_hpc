#include <mkl.h>
#include <vector.h>
#include <stdio.h>

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

double* malloc_vector(size_t size, int i){
    return (double*) mkl_malloc(size, i);
}

void vector_free(Vector vec){
    if(vec.data != NULL){
        free(vec.data);
    }
    return;
}

double* add_vector(Vector x, Vector y){
    double* a = malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, y.data, 1, a, 1);
    cblas_daxpy(x.length, 1, x.data, 1, a, 1);
    return a;
}

void iadd_vector(Vector x, Vector y){
    cblas_daxpy(x.length, 1.0, y.data, 1, x.data, 1);
}

void iadd_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, x.data);
}


double *add_scalar(Vector x, double scal){
    double* a = malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return a;
}

void mul_vector(Vector x, Vector y, double *res){
    vdMul(x.length, x.data, y.data, res);
    return;
}
void mul_scalar(Vector x, double scal, double *res){
    vdLinearFrac(x.length, x.data, x.data, scal , 0, 0, 1, res);
    return;
}

void imul_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, scal , 0, 0, 1, x.data);
    return;
}
void imul_vector(Vector x, Vector y){
    vdMul(x.length, x.data, y.data, x.data);
    return;
}

double* sub_vector(Vector x, Vector y){
    double* a = malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, y.data, 1, a, 1);
    cblas_daxpy(x.length, -1.0, x.data, 1, a, 1);
    return a;
}

void isub_vector(Vector x, Vector y){
    cblas_daxpy(x.length, -1.0, y.data, 1, x.data, 1);
}

void isub_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, -scal, 0, 1, x.data);
}


double *sub_scalar(Vector x, double scal){
    double* a = malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 1, -scal, 0, 1, a);
    return a;
}

double* rsub_vector(Vector x, Vector y){
    double* a = malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, y.data, 1, a, 1);
    cblas_daxpy(x.length, -1.0, x.data, 1, a, 1);
    return a;
}

double *rsub_scalar(Vector x, double scal){
    double* a = malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, -1, scal, 0, 1, a);
    return a;
}

double *pow_vector(Vector x, Vector y){
    double* a = malloc(x.length * sizeof(double), 512);
    vdPow(x.length, x.data, y.data, a);
    return a;
}

double *pow_scalar(Vector x, double scal){
    double* a = malloc(x.length * sizeof(double), 512);
    vdPowx(x.length, x.data, scal, a);
    return a;
}

double *truediv_vector(Vector x, Vector y){
    double* a = malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, x.data, y.data, a);
    return a;
}

double *truediv_scalar(Vector x, double scal){
    double* a = malloc(x.length * sizeof(double), 512);
    mkl.cblas_daxpby(x.length, 1/scal, x.data, 1, 0.0, a, 1);
    return a;
}

double *rtruediv_vector(Vector x, Vector y){
    double* a = malloc(x.length * sizeof(double), 512);
    vdDiv(x.length, y.data, x.data, a);
    return a;
}

double *rtruediv_scalar(Vector x, double scal){
    double* a = malloc(x.length * sizeof(double), 512);
    vdInv(x.length, x.data, a);
    mkl.cblas_dscal(x.length, scal, a, 1);
    return a;
}

void itruediv_vector(Vector x, Vector y){
    vdDiv(x.length, x.data, y.data, x.data);
    return a;
}

void itruediv_scalar(Vector x, double scal){
    mkl.cblas_daxpby(x.length, 1/scal, x.data, 1, 0.0, x.data, 1);
    return a;
}

double *neg(Vector x){
    double* a = malloc(x.length * sizeof(double), 512);
    cblas_daxpby(x.length, -1, x.data, 1, 0, a, 1);
    return a;
}
