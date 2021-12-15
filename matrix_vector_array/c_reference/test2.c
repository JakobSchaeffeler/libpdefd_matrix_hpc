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

	double* y_a_tmp = rand_arr_omp(N);
	double* x_a_tmp = rand_arr_omp(N);	

	vdPow(N, x_a_tmp, y_a_tmp, x_a_tmp);
    	mkl_free(x_a_tmp);
    	mkl_free(y_a_tmp);

}





