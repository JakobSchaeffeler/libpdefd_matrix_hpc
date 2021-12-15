#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vector_blas.h"
#include "vector_vml.h"
#include <string.h>
#include <omp.h>
#include <unistd.h>

double* rand_arr(int length){
    	double* x = (double*) mkl_malloc(length * sizeof(double), 64);	
	for(int i=0;i<length;i++)
        	 x[i]= rand();
    	
	return x;
}
double* rand_arr_omp(int length){
    	double* x = (double*) mkl_malloc(length * sizeof(double), 64);	
#pragma omp parallel for
	for(int i=0;i<length;i++)
        	 x[i]= rand();
    	
	return x;
}
int main(){
    	int min_time = 1;
    	int max_cores = 24;
	double* x_a = NULL;
    	double* x_b = NULL;

    	double* y_a = NULL;
    	double* y_b = NULL;

    	Vector_blas x_blas;
    	Vector_blas y_blas;
        int N = 16000000;

  	int k = 0;
    	struct timespec start;
    	struct timespec end;
    	double time_overall = 0.0;
	double time_min = 10000000;
	double time_max = -1.0;
	double diff;
    	for(int cores = 1; cores <= max_cores; cores++){	
		mkl_set_num_threads(cores);
		mkl_free_buffers();
		omp_set_num_threads(cores);
	    	k = 0;
		mkl_free(x_a);
		mkl_free(y_a);
		x_a = (double *) mkl_malloc(N * sizeof(double), 64);
		y_a = (double *) mkl_malloc(N * sizeof(double), 64);
	
		double* y_a_tmp = rand_arr_omp(N);
		double* x_a_tmp = rand_arr_omp(N);	

		double *x_calloc = (double *) mkl_calloc(N, sizeof(double), 64);
		double *y_calloc = (double *) mkl_calloc(N, sizeof(double), 64);
		double *x_pby = (double *) mkl_malloc(N * sizeof(double), 64);
                double *y_pby = (double *) mkl_malloc(N * sizeof(double), 64);

		cblas_daxpy(N, 1.0, x_a_tmp, 1, x_calloc, 1);
                cblas_daxpy(N, 1.0, y_a_tmp, 1, y_calloc, 1);
		cblas_daxpby(N, 1.0, y_a_tmp, 1, 0.0, y_pby, 1);
		cblas_daxpby(N, 1.0, x_a_tmp, 1, 0.0, x_pby, 1);

                for(int i = 0; i < 10;i++){
                        cblas_daxpy(N, 1.0, y_pby, 1, x_pby, 1);

                }



		for(int i = 0; i < 10;i++){
                        cblas_daxpy(N, 1.0, y_calloc, 1, x_calloc, 1);

		}

			
		for(int i = 0; i < 10; i++){
			cblas_daxpy(N, 1.0, y_a_tmp, 1, x_a_tmp, 1);
		}

		cblas_dcopy(N, x_a_tmp, 1, x_a, 1);
		cblas_dcopy(N, y_a_tmp, 1, y_a, 1);
		x_blas = vector_new_blas(x_a, N);
		y_blas = vector_new_blas(y_a, N);
	    	mkl_free(x_a_tmp);
	    	mkl_free(y_a_tmp);
	        x_a_tmp = NULL;
		y_a_tmp = NULL;	
	    	double* res = NULL;
	

		while(time_overall < min_time){
			k += 1;
		    	clock_gettime(CLOCK_MONOTONIC, &start);
		    	iadd_vector_blas(x_blas, y_blas);
		    	clock_gettime(CLOCK_MONOTONIC, &end);
		    	diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
		    	//mkl_free(res);
		    	time_overall += diff;
		}
		mkl_free(x_a);
		mkl_free(y_a);
		x_a = NULL;
		y_a = NULL;
		printf("%d iterations (BLAS) took %f seconds, %f seconds per operation\n", k, time_overall, time_overall/k);
		k = 0;
		diff = 0.0;
		time_overall = 0.0;
	}
	
}





