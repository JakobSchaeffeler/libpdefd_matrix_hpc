#include "vector.hpp"

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
        mkl_free(vec.data);
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

double* alloc_vector(int size){

    double* a = MemBlockAlloc::alloc<double>(size * sizeof(double));
    if(a == NULL){
        printf("alloc failed");
    }
    return a;

}

void alloc_free(double* a, int size){
    MemBlockAlloc::free(a, size * sizeof(double));
    return;
}

double* malloc_vector(int size){
    double* a = (double*) malloc(size * sizeof(double));
    if(a == NULL){
        printf("Malloc failed");
    }
    return a;
}

void ptr_free(double* x){
    if(x != NULL){
        free(x);
        x = NULL;
    }
}

double* mkl_malloc_vector(int size){
    double* a = (double*) mkl_malloc(size * sizeof(double), 512);
    if(a == NULL){
        printf("Malloc failed");
    }
    return a;
}

double* mkl_realloc_vector(double* a, int size){
    double* b = (double*) mkl_realloc(a, size * sizeof(double));
    if(b == NULL){
        printf("realloc failed");
    }
    return b;
}

void ptr_mkl_free(double* x){
    if(x != NULL){
        mkl_free(x);
        x = NULL;
    }
}

void vector_free(Vector vec){
    if(vec.data != NULL){
        MemBlockAlloc::free(vec.data, vec.length * sizeof(double));
        vec.data = NULL;
    }
    return;
}

double* numa_alloc_vector(int size){
    double* a = (double*) numa_alloc(size * sizeof(double));
    return a;
}
void numa_free_vector(double* a, int size){
    numa_free(a, size * sizeof(double));
    return;
}


Vector add_vector(Vector x, Vector y){

    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_dcopy(x.length, y.data, 1, a, 1);
    cblas_daxpy(x.length, 1.0, x.data, 1, a, 1);
    return vector_new(a, x.length);
}

Vector add_scalar(Vector x, double scal){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return vector_new(a, x.length);
}


void print_affinity() {
    cpu_set_t mask;
    long nproc, i;

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        assert(false);
    }
    nproc = sysconf(_SC_NPROCESSORS_ONLN);
    printf("sched_getaffinity = ");
    for (i = 0; i < nproc; i++) {
        printf("%d ", CPU_ISSET(i, &mask));
    }
    printf("\n");
}



void iadd_vector(Vector x, Vector y){
    cblas_daxpy(x.length, 1, y.data, 1, x.data, 1);
}

void iadd_double_vector(double* x, double* y, int n){
    cblas_daxpy(n, 1, y, 1, x, 1);
}

void iadd_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, x.data);
}

Vector mul_vector(Vector x, Vector y){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_dsbmv(CblasRowMajor, CblasLower, x.length, 0, 1.0, x.data, 1, y.data, 1,
    0.0, a, 1);
    return vector_new(a, x.length);
}
Vector mul_scalar(Vector x, double scal){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_daxpby(x.length, scal, x.data, 1, 0.0, a, 1);
    return vector_new(a, x.length);
}

void imul_scalar(Vector x, double scal){
    cblas_dscal(x.length, scal, x.data, 1);
    return;
}

void imul_double_scalar(double* x, double scal, int n){
    cblas_dscal(n, scal, x, 1);
    return;
}

void imul_vector(Vector x, Vector y){
    cblas_dsbmv(CblasRowMajor, CblasLower, x.length, 0, 1.0, x.data, 1, y.data, 1, 0.0, x.data, 1);
    return;
}

Vector sub_vector(Vector x, Vector y){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_dcopy(x.length, x.data, 1, a, 1);
    cblas_daxpy(x.length, -1, y.data, 1, a, 1);
    return vector_new(a, x.length);
}

Vector sub_scalar(Vector x, double scal){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdLinearFrac(x.length, x.data, x.data, 1, -scal, 0, 1, a);
    return vector_new(a, x.length);
}

void isub_vector(Vector x, Vector y){
    cblas_daxpy(x.length, -1.0, y.data, 1, x.data, 1);
}

void isub_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, -scal, 0, 1, x.data);
}

Vector rsub_vector(Vector x, Vector y){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_dcopy(x.length, y.data, 1, a, 1);
    cblas_daxpy(x.length, -1.0, x.data, 1, a, 1);
    return vector_new(a, x.length);
}

Vector rsub_scalar(Vector x, double scal){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdLinearFrac(x.length, x.data, x.data, -1, scal, 0, 1, a);
    return vector_new(a, x.length);
}

Vector pow_vector(Vector x, Vector y){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdPow(x.length, x.data, y.data, a);
    return vector_new(a, x.length);
}

Vector pow_scalar(Vector x, double scal){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdPowx(x.length, x.data, scal, a);
    return vector_new(a, x.length);
}

Vector truediv_vector(Vector x, Vector y){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdDiv(x.length, x.data, y.data, a);
    return vector_new(a, x.length);
}

Vector truediv_scalar(Vector x, double scal){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_daxpby(x.length, 1/scal, x.data, 1, 0.0, a, 1);
    return vector_new(a, x.length);
}

Vector rtruediv_vector(Vector x, Vector y){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdDiv(x.length, y.data, x.data, a);
    return vector_new(a, x.length);
}

Vector rtruediv_scalar(Vector x, double scal){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    vdInv(x.length, x.data, a);
    cblas_dscal(x.length, scal, a, 1);
    return vector_new(a, x.length);
}

void itruediv_vector(Vector x, Vector y){
    vdDiv(x.length, x.data, y.data, x.data);
    return;
}

void itruediv_scalar(Vector x, double scal){
    cblas_daxpby(x.length, 1/scal, x.data, 1, 0.0, x.data, 1);
    return;
}

Vector neg(Vector x){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_daxpby(x.length, -1, x.data, 1, 0.0, a, 1);
    return vector_new(a, x.length);
}

Vector copy(Vector x){
    double* a = MemBlockAlloc::alloc<double>(x.length * sizeof(double));
    cblas_dcopy(x.length, x.data, 1, a, 1);
    return vector_new(a, x.length);
}

void copy_double(double* x, double* y, int n){
    cblas_dcopy(n, x, 1, y, 1);
    return;
}

Vector vec_from_ptr(int length, double* data){
    double* res = MemBlockAlloc::alloc<double>(length * sizeof(double));
    cblas_dcopy(length, data, 1, res, 1);
    Vector newvector;
    newvector.data = res;
    newvector.length = length;
    return newvector;
}

void set_threads(int i){
    mkl_set_num_threads(i);
}

