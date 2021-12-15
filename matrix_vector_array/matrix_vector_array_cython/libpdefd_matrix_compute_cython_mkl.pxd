cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse as sparse
import sys, os

a, trash = os.path.split(os.path.abspath(__file__))

path = os.path.join(a, 'mkl_typed')
sys.path.append(path)

from libpdefd_vector_array_cython_mkl_typed cimport vector_array_base as vector_array_base_typed
sys.path.pop()
"""
path = os.path.join(a, 'mkl_naiv')
sys.path.append(path)
from libpdefd_vector_array_cython_mkl_naiv cimport vector_array_base as vector_array_base_naiv
sys.path.pop()
"""
path = os.path.join(a, 'mkl_malloc')
sys.path.append(path)
from libpdefd_vector_array_cython_mkl_malloc cimport vector_array_base as vector_array_base_malloc
sys.path.pop()
path = os.path.join(a, 'mkl_cpp_alloc')
sys.path.append(path)
from libpdefd_vector_array_cython_mkl_cpp_alloc cimport vector_array_base as vector_array_base_cpp_alloc
sys.path.pop()


a, trash = os.path.split(a)
sys.path.append(a)
import libpdefd_matrix_setup




"""
This module implements the compute class for sparse matrices

Hence, this should be highly optimized
"""

cdef extern from "mkl_spblas.h":

    ctypedef enum sparse_matrix_type_t:
        SPARSE_MATRIX_TYPE_GENERAL            = 20
        SPARSE_MATRIX_TYPE_SYMMETRIC          = 21
        SPARSE_MATRIX_TYPE_HERMITIAN          = 22
        SPARSE_MATRIX_TYPE_TRIANGULAR         = 23
        SPARSE_MATRIX_TYPE_DIAGONAL           = 24
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR   = 25
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL     = 26

    ctypedef enum sparse_fill_mode_t:
        SPARSE_FILL_MODE_LOWER  = 40
        SPARSE_FILL_MODE_UPPER  = 41
        SPARSE_FILL_MODE_FULL   = 42

    ctypedef enum sparse_diag_type_t:
        SPARSE_DIAG_NON_UNIT    = 50
        SPARSE_DIAG_UNIT        = 51


    cdef struct matrix_descr:
        sparse_matrix_type_t type
        sparse_fill_mode_t mode
        sparse_diag_type_t diag

    struct  sparse_matrix:
        pass

    ctypedef sparse_matrix* sparse_matrix_t

    ctypedef enum sparse_operation_t:
        SPARSE_OPERATION_NON_TRANSPOSE = 10
        SPARSE_OPERATION_TRANSPOSE = 11
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12

# class _sparse_matrix(Structure):
#     pass
#
#
# cdef class matrix_descr(Structure):
#     _fields_ = [("sparse_matrix_type_t", c_int),
#                 ("sparse_fill_mode_t", c_int),
#                 ("sparse_diag_type_t", c_int)]
#
#     def __init__(self, sparse_matrix_type_t=20, sparse_fill_mode_t=0, sparse_diag_type_t=0):
#         super(matrix_descr, self).__init__(sparse_matrix_type_t, sparse_fill_mode_t, sparse_diag_type_t)
#
#
# matr_dec = matrix_descr()

cdef extern from "mkl.h":

    ctypedef int MKL_INT

    ctypedef enum sparse_status_t:
        SPARSE_STATUS_SUCCESS = 0
        SPARSE_STATUS_NOT_INITIALIZED = 1
        SPARSE_STATUS_ALLOC_FAILED = 2
        SPARSE_STATUS_INVALID_VALUE = 3
        SPARSE_STATUS_EXECUTION_FAILED = 4
        SPARSE_STATUS_INTERNAL_ERROR = 5
        SPARSE_STATUS_NOT_SUPPORTED = 6

    ctypedef enum sparse_index_base_t:
        SPARSE_INDEX_BASE_ZERO = 0
        SPARSE_INDEX_BASE_ONE = 1



    cdef sparse_status_t mkl_sparse_d_create_csr (sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, double *values);
    cdef sparse_status_t mkl_sparse_copy( const sparse_matrix_t source, const matrix_descr descr, sparse_matrix_t* dest)
    cdef sparse_status_t mkl_sparse_d_add( const sparse_operation_t operation, const sparse_matrix_t A, const double alpha,
                                           const sparse_matrix_t B, sparse_matrix_t *C)
    cdef sparse_status_t mkl_sparse_destroy(sparse_matrix_t A)
    cdef void* mkl_malloc(size_t len, int n) nogil
    cdef void* mkl_free(void* ptr) nogil
    cdef sparse_status_t mkl_sparse_d_mv (const sparse_operation_t  operation, const double alpha, const sparse_matrix_t A,
                                          const matrix_descr descr, const double *x, const double beta, double *y )
    cdef sparse_status_t mkl_sparse_d_export_csr(const sparse_matrix_t source, sparse_index_base_t *indexing, MKL_INT *rows,
                                                 MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end, MKL_INT **col_indx,
                                                 double **values)


cdef class matrix_sparse:
    cdef sparse_matrix_t _matrix_mkl
    cdef tuple shape
    cdef int shape_len
    cdef int[:] indptr
    cdef int[:] indices
    cdef double[:] values
    cdef setup(self, data, tuple shape)
    cdef setup_empty(self, tuple shape)
    cdef iadd(self, matrix_sparse data)
    #cdef add(self, matrix_sparse data)
    #cdef vector_array_base_naiv dot_add_reshape_array_base_mkl(self, vector_array_base_naiv x, vector_array_base_naiv c, tuple dst_shape)
    cdef vector_array_base_typed dot_add_reshape_array_base_mkl_typed(self, vector_array_base_typed x, vector_array_base_typed c, tuple dst_shape)
    cdef vector_array_base_malloc dot_add_reshape_array_base_mkl_malloc(self, vector_array_base_malloc x, vector_array_base_malloc c, tuple dst_shape)
    cdef vector_array_base_cpp_alloc dot_add_reshape_array_base_mkl_cpp_alloc(self, vector_array_base_cpp_alloc x, vector_array_base_cpp_alloc c, tuple dst_shape)

