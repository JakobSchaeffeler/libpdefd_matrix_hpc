import os
import sys
import importlib
import pyximport
pyximport.install(reload_support=True, language_level=3)

"""
Use this hack to use these python files also without libpdefd
"""
import sys, os
a, trash = os.path.split(os.path.dirname(__file__))
path = os.path.join(a, 'matrix_vector_array')
sys.path.append(path)

cy_path = os.path.join(path, 'matrix_vector_array_cython')
sys.path.append(cy_path)
import libpdefd_matrix_compute_mkl as libpdefd_matrix_compute_mkl_cython
cimport libpdefd_matrix_compute_cython_mkl as libpdefd_matrix_compute_mkl_cython
sys.path.append(os.path.join(cy_path, "mkl_blas_c"))
import libpdefd_vector_array_cython_mkl_blas_c as libpdefd_vector_array_cython_mkl_blas_c
cimport libpdefd_vector_array_cython_mkl_blas_c as libpdefd_vector_array_cython_mkl_blas_c
sys.path.pop()
sys.path.pop()

def add(x, y, int k):
    for _ in range(k):
        x.__add__(y)
    return

def add_scalar(x, double y, int k):
    for _ in range(k):
        x.add_scalar(y)
    return


def add_vector(x,y, int k):
    for _ in range(k):
        x.add_vector(y)
    return


def add_matrix_cython(x,y,int k):
    for _ in range(k):
        x.iadd(y)
    return

def add_matrix(x,y,int k):
    for _ in range(k):
        x.__iadd__(y)
    return

def dot_add_reshape(A, x, y, dst_shape, int k):
    for _ in range(k):
        A.dot_add_reshape(x,y,dst_shape)
    return

def dot_add_reshape_cython(A,  libpdefd_vector_array_cython_mkl_blas_c.vector_array_base x,
                           libpdefd_vector_array_cython_mkl_blas_c.vector_array_base y, tuple dst_shape, int k):
    for _ in range(k):
        A.dot_add_reshape(x,y,dst_shape)


