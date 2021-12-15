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


int main(int argc, char**argv){
	if(argc != 3){
		printf("Wrong number of args used\n");
		return
	}
	int N = atoi(argv[2]);
    	int min_time = 5;
    	int cores = atoi(argv[1]);
	int len=198;
  	char buf[198];
  	mkl_get_version_string(buf, len);
  	printf("%s\n",buf);
    	double* x_a = NULL;
    	double* x_b = NULL;

    	double* y_a = NULL;
    	double* y_b = NULL;

    	Vector_blas x_blas;
    	Vector_blas y_blas;

    	Vector_vml x_vml;
    	Vector_vml y_vml;

    	//double* res;
	char str_N[10];
	sprintf(str_N, "%d", N);
	printf("%s\n", str_N);

    	int k = 0;
    	struct timespec start;
    	struct timespec end;
    	double time_overall = 0.0;
	double time_min = 10000000;
	double time_max = -1.0;
	double diff;
	char str_f[40];
	char str_f_min[40];
	char str_f_max[40];
	FILE *f; 
	FILE *f_min; 
	FILE *f_max;
	sprintf(str_f, "blas_c_iadd_%d_%d.csv", N, cores);
	sprintf(str_f_min, "blas_c_iadd_%d_%d_min.csv", N, cores);
	sprintf(str_f_max, "blas_c_iadd_%d_%d_max.csv", N, cores);
	x_a = NULL;
	x_b = NULL;
   	y_a = NULL;
    	y_b = NULL;
 			
	f = fopen(str_f, "w"); 
	f_min = fopen(str_f_min, "w"); 
	f_max = fopen(str_f_max, "w");
	if (f == NULL) return -1; 
	mkl_set_num_threads(cores);
	omp_set_num_threads(cores);
	k = 0;
	mkl_free(x_a);
	mkl_free(y_a);
	x_a = (double *) mkl_malloc(N * sizeof(double), 64);
	y_a = (double *) mkl_malloc(N * sizeof(double), 64);
	printf("Hi\n");
	double* y_a_tmp = rand_arr(N);
	double* x_a_tmp = rand_arr(N);
	printf("pragma ops\n");
	cblas_dcopy(N, x_a_tmp, 1, x_a, 1);
	cblas_dcopy(N, y_a_tmp, 1, y_a, 1);
	x_blas = vector_new_blas(x_a, N);
	y_blas = vector_new_blas(y_a, N);
	mkl_free(x_a_tmp);
	mkl_free(y_a_tmp);
	x_a_tmp = NULL;
	y_a_tmp = NULL;	
	double* res = NULL;
				//warmup

	for(int i = 0; i < 10; i++){
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_vector_blas(x_blas, y_blas);
		clock_gettime(CLOCK_MONOTONIC, &end);
		//mkl_free(res);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
	}

	while(time_overall < min_time){
		k += 1;
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_vector_blas(x_blas, y_blas);
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
		//mkl_free(res);
		if(diff < time_min){
		time_min = diff;
		}
		if(diff > time_max){
		time_max = diff;
		}
		time_overall += diff;
	}

	mkl_free(x_a);
	mkl_free(y_a);
	x_a = NULL;
	y_a = NULL;
	printf("%d iterations (BLAS) took %f seconds, %f seconds per operation\n", k, time_overall, time_overall/k);
	fprintf(f, "%.20f\n", time_overall/k);
	fprintf(f_min, "%.20f\n", time_min);
	fprintf(f_max, "%.20f\n", time_max);
	k = 0;
	diff = 0.0;
	time_overall = 0.0;
	time_min = 10000000;
	time_max = -1.0;
	
	fclose(f);
	fclose(f_min);
	fclose(f_max);


	sprintf(str_f, "blas_c_iadd_%d_scalar.csv", N);
	sprintf(str_f_min, "blas_c_iadd_%d_scalar_min.csv", N);
	sprintf(str_f_max, "blas_c_iadd_%d_scalar_max.csv", N);
					
	f = fopen(str_f, "w"); 
	f_min = fopen(str_f_min, "w"); 
	f_max = fopen(str_f_max, "w"); 



	if (f == NULL) return -1; 
	x_a = mkl_malloc(N * sizeof(double), 64);
	double* x_a_tmp = rand_arr(N);
	double a=rand();
	cblas_dcopy(N, x_a_tmp, 1, x_a, 1);
	x_blas = vector_new_blas(x_a, N);
	mkl_free(x_a_tmp);
	double* res = NULL;
	//warmup
	for(int i = 0; i < 10; i++){
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_scalar_blas(x_blas, a);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mkl_free(res);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
	}
	while(time_overall < min_time){
		k += 1;
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_scalar_blas(x_blas, a);
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
		mkl_free(res);
		if(diff < time_min){
			time_min = diff;
		}
		if(diff > time_max){
			time_max = diff;
		}
		time_overall += diff;
	}
	printf("%d iterations (BLAS SCALAR) took %f seconds, %f seconds per operation\n", k, time_overall, time_overall/k);
	mkl_free(x_a);
	fprintf(f, "%.20f\n", time_overall/k);
	fprintf(f_min, "%.20f\n", time_min);
	fprintf(f_max, "%.20f\n", time_max);
	k = 0;
	diff = 0.0;
	time_overall = 0.0;
	time_min = 10000000;
	time_max = -1.0;
	

	fclose(f);
	fclose(f_min);
	fclose(f_max);
	sprintf(str_f, "vml_c_iadd_%d.csv", N);
	sprintf(str_f_min, "vml_c_iadd_%d_min.csv", N);
	sprintf(str_f_max, "vml_c_iadd_%d_max.csv", N);

	f = fopen(str_f, "w"); 
	f_min = fopen(str_f_min, "w"); 
	f_max = fopen(str_f_max, "w"); 


	if (f == NULL) return -1; 

	mkl_set_num_threads(cores);
	k = 0;
	x_a = mkl_malloc(N * sizeof(double), 64);
	y_a = mkl_malloc(N * sizeof(double), 64);
	double* y_a_tmp = rand_arr(N);
	double* x_a_tmp = rand_arr(N);
	cblas_dcopy(N, x_a_tmp, 1, x_a, 1);
	cblas_dcopy(N, y_a_tmp, 1, y_a, 1);
	x_vml = vector_new_vml(x_a, N);
	y_vml = vector_new_vml(y_a, N);
	mkl_free(x_a_tmp);
	mkl_free(y_a_tmp);
	//warmup
	double* res = NULL;
	for(int i = 0; i < 10; i++){
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_vector_vml(x_vml, y_vml);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mkl_free(res);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
	}
			
	while(time_overall < min_time){
		k += 1;
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_vector_vml(x_vml, y_vml);
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
		mkl_free(res);
		if(diff < time_min){
			time_min = diff;
		}
		if(diff > time_max){
			time_max = diff;
		}
		time_overall += diff;
	}
	printf("%d iterations (VML) took %f seconds, %f seconds per operation\n", k, time_overall, time_overall/k);
	fprintf(f, "%.20f\n", time_overall/k);
	fprintf(f_min, "%.20f\n", time_min);
	fprintf(f_max, "%.20f\n", time_max);
	k = 0;
	diff = 0.0;
	time_overall = 0.0;
	time_min = 10000000;
	time_max = -1.0;
	mkl_free(x_a);
	mkl_free(y_a);

	    
	fclose(f);
	fclose(f_min);
	fclose(f_max);
	sprintf(str_f, "vml_c_iadd_%d_scalar.csv", N);
	sprintf(str_f_min, "vml_c_iadd_%d_scalar_min.csv", N);
	sprintf(str_f_max, "vml_c_iadd_%d_scalar_max.csv", N);

	f = fopen(str_f, "w"); 
	f_min = fopen(str_f_min, "w"); 
	f_max = fopen(str_f_max, "w"); 


	if (f == NULL) return -1; 
	x_a = mkl_malloc(N * sizeof(double), 64);
	double* x_a_tmp = rand_arr(N);
	double a = rand();
	cblas_dcopy(N, x_a_tmp, 1, x_a, 1);
	x_vml = vector_new_vml(x_a, N);
	mkl_free(x_a_tmp);
	double* res = NULL;
	//warmup
	for(int i = 0; i < 10; i++){
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_scalar_vml(x_vml,a);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mkl_free(res);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
	}
	while(time_overall < min_time){
		k += 1;
		clock_gettime(CLOCK_MONOTONIC, &start);
		iadd_scalar_vml(x_vml, a);
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0e-9 * (end.tv_nsec - start.tv_nsec);
		mkl_free(res);
		if(diff < time_min){
			time_min = diff;
		}
		if(diff > time_max){
			time_max = diff;
		}
		time_overall += diff;
	}
	printf("%d iterations (VML SCALAR) took %f seconds, %f seconds per operation\n", k, time_overall, time_overall/k);
	mkl_free(x_a);
	fprintf(f, "%.20f\n", time_overall/k);
	fprintf(f_min, "%.20f\n", time_min);
	fprintf(f_max, "%.20f\n", time_max);
	k = 0;
	diff = 0.0;
	time_overall = 0.0;
	time_min = 10000000;
	time_max = -1.0;
	}

	fclose(f);
	fclose(f_min);
	fclose(f_max);

}





