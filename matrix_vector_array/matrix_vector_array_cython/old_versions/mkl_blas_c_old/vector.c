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

double* malloc_vector(size_t size){
    return (double*) mkl_malloc(size, 512);
}

void ptr_free(double* x){
    if(x != NULL){
        mkl_free(x);
        x = NULL;
    }
}

void vector_free(Vector vec){
    if(vec.data != NULL){
        mkl_free(vec.data);
        vec.data = NULL;
    }
    return;
}


double* add_vector(Vector x, Vector y){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, y.data, 1, a, 1);
    cblas_daxpy(x.length, 1.0, x.data, 1, a, 1);
    return a;
}

double *add_scalar(Vector x, double scal){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, a);
    return a;
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
    /*#pragma omp parallel for
    for(int i=0; i < x.length; i++){
        x.data[i] += y.data[i];
        int nThreads=omp_get_num_threads();
        printf("nThreads: %d\n",nThreads);
    }

    print_affinity();
    printf("sched_getcpu = %d\n", sched_getcpu());
    }
    */
    cblas_daxpy(x.length, 1.0, y.data, 1, x.data, 1);
}

void iadd_scalar(Vector x, double scal){
    vdLinearFrac(x.length, x.data, x.data, 1, scal, 0, 1, x.data);
}

double* mul_vector(Vector x, Vector y){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    cblas_dsbmv(101, 122, x.length, 0, 1.0, x.data, 1, y.data, 1,
    0.0, a, 1);
    return a;
}
double* mul_scalar(Vector x, double scal){
    double* a = (double*) mkl_malloc(x.length * sizeof(double), 512);
    cblas_daxpby(x.length, scal, x.data, 1, 0.0, a, 1);
    return a;
}

void imul_scalar(Vector x, double scal){
    cblas_dscal(x.length, scal, x.data, 1);
    return;
}
void imul_vector(Vector x, Vector y){
    cblas_dsbmv(101, 122, x.length, 0, 1.0, x.data, 1, y.data, 1, 0.0, x.data, 1);
    return;
}

Vector copy(Vector x){
    double* a = mkl_malloc(x.length * sizeof(double), 512);
    cblas_dcopy(x.length, x.data, 1, a, 1);
    return vector_new(a,x.length);
}

Vector vec_from_ptr(int length, double* data){
    double* res = mkl_malloc(length * sizeof(double), 512);
    cblas_dcopy(length, data, 1, res, 1);
    Vector newvector;
    newvector.data = res;
    newvector.length = length;
    return newvector;
}


