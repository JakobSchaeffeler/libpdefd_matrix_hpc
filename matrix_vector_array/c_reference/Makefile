CFLAGS=-O3 -g  
CC = icc

.Phony: all
all: main

main: benchmarks.c vector_blas.c vector_vml.c
	#$(CC) $(CFLAGS) -o $@ $^  -L$(CONDA_PREFIX)/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -DMKL_ILP64  -I$(CONDA_PREFIX)/include -qopenmp
	
	
	#$(CC) $(CFLAGS) -o $@ $^ -L$(CONDA_PREFIX)/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -lmkl_rt  -DMKL_ILP64  -m64  -I"${CONDA_PREFIX}/include" -fopenmp
	
	#$(CC) $(CFLAGS) -o $@ $^ -L$(CONDA_PREFIX)/lib -liomp5 -lpthread -lm -ldl -lmkl_rt -DMKL_ILP64 -I"${CONDA_PREFIX}/include" -mkl=parallel 

	$(CC) $(CFLAGS) -o $@ $^  -L$(CONDA_PREFIX)/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -I$(CONDA_PREFIX)/include -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -DMKL_ILP64 -qopenmp 
	
	#$(CC) $(CFLAGS) -o $@ $^  -L$(CONDA_PREFIX)/lib  -I$(CONDA_PREFIX)/include -qopenmp -L/shared/oneapi/tbb/2021.3.0/lib/intel6/ -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lpthread -lm -ldl  -DMKL_ILP64 -L/shared/intel/2020/intelpython3/lib/ 
	


	#-L/shared/intel/2020/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64 -L/shared/intel/2020/compilers_and_libraries_2020.2.254/linux/mkl/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -I/shared/intel/2020/compilers_and_libraries_2020.2.254/linux/mkl/include -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -DMKL_ILP64 -qopenmp 
