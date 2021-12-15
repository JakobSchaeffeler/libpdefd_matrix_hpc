#include <mkl.h>

typedef struct Vector{
    int length;
    double* data;
};


Vector vector_new(double* data, int length){
    struct vector newvector;
    newvector.data = data;
    newvector.length = length;
    return newvector;

}

void vector_free(Vector vec){
    mkl_free(vec.data);
    return;
}

Vector add_vector(Vector x, Vector y){
    double *a = (double *) mkl_malloc(x.length * sizeof(double), 64);
    vdAdd(x.length, x.data, y.data, a);
    return vector_new(x.length, a);
}

Vector add_scalar(Vector x, double scal){
    double *a = (double *) mkl_malloc(x.length * sizeof(double), 64);
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return vector_new(x.length, a);

}
