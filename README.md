# libpdefd_matrix_hpc

## Requirements:
- Set up MKL environment (MKLROOT needs to be set)
- Intel C++ Compiler (icpc)


## Initial Setup
1. run ./install\_anaconda.sh
2. run ./python\_setup.sh
3. run source python\_venv\_anaconda\_*name*/bin/activate
4. run ./compile\_cython\_versions.sh (some compilations fail initially but this is normal behaviour)

## Settings required (need to be executed each time)
1. run run source python\_venv\_anaconda\_*name*/bin/activate
2. Set environment variables via . ./set\_environment.sh
3. Select a version to use by setting PYSM_BACKEND to either *scipy*, *mkl*, *mkl_cython_naiv*, *mkl_cython_typed*, *mkl_cython_malloc*, *mkl_cython_cpp* 


## Versions
- *scipy*: Base implementation using only NumPy/SciPy
- *mkl*: Implementation using MKL in Python
- *mkl_cython_naiv*: Implementation using MKL in Cython with Python Code
- *mkl_cython_typed*: Implementation using MKL in Cython with type annotations
- *mkl_cython_malloc*: Implementation using MKL in Cython with type annotations and *mkl_malloc*
- *mkl_cython_cpp*:  Implementation using MKL in Cython with Block Allocatior and all functions in C++
