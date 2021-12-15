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


cdef extern from "matrix.hpp":
    sparse_matrix_t add(sparse_matrix_t m_a, sparse_matrix_t m_b);
    void dot_add(sparse_matrix_t m, double* x, double* y);
