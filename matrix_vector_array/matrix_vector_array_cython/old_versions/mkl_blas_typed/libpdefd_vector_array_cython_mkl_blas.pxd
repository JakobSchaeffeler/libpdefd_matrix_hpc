import numpy as np
cimport numpy as np

cdef class vector_array_base:
    cdef double* _data_as
    cdef public int length
    cdef public tuple shape
    cdef vector_array_base add_vector(self, vector_array_base a)
    cdef vector_array_base add_scalar(self, double a)
    cdef void iadd_vector(self, vector_array_base a)
    cdef void iadd_scalar(self, double a)
    cdef vector_array_base sub_vector(self, vector_array_base a)
    cdef vector_array_base sub_scalar(self, double a)
    cdef vector_array_base rsub_vector(self, vector_array_base a)
    cdef vector_array_base rsub_scalar(self, double a)
    cdef void isub_vector(self, vector_array_base a)
    cdef void isub_scalar(self, double a)
    cdef vector_array_base mul_vector(self, vector_array_base a)
    cdef vector_array_base mul_scalar(self, double a)
    cdef void imul_vector(self, vector_array_base a)
    cdef void imul_scalar(self, double a)
