import numpy as np
cimport numpy as np

cdef class vector_array_base:
    cdef public int length
    cdef public tuple shape
    cdef double* data_as
    cdef vector_array_base add_vector(vector_array_base self, vector_array_base b)
    cdef vector_array_base add_scalar(vector_array_base self, double b)
    cdef void iadd_vector(vector_array_base self, vector_array_base b)
    cdef void iadd_scalar(vector_array_base self, double b)
    cdef vector_array_base sub_scalar(vector_array_base self, double a)
    cdef vector_array_base sub_vector(vector_array_base self, vector_array_base a)
    cdef void isub_scalar(vector_array_base self, double a)
    cdef void isub_vector(vector_array_base self, vector_array_base a)
    cdef vector_array_base rsub_scalar(vector_array_base self, double a)
    cdef vector_array_base rsub_vector(vector_array_base self, vector_array_base a)
    cdef vector_array_base mul_scalar(vector_array_base self, double a)
    cdef vector_array_base mul_vector(vector_array_base self, vector_array_base a)
    cdef void imul_scalar(vector_array_base self, double a)
    cdef void imul_vector(vector_array_base self, vector_array_base a)
    cdef vector_array_base pow_scalar(vector_array_base self, double a)
    cdef vector_array_base pow_vector(vector_array_base self, vector_array_base a)
    cdef vector_array_base truediv_scalar(vector_array_base self, double a)
    cdef vector_array_base truediv_vector(vector_array_base self, vector_array_base a)
    cdef vector_array_base rtruediv_scalar(vector_array_base self, double a)
    cdef vector_array_base rtruediv_vector(vector_array_base self, vector_array_base a)
    cdef void itruediv_scalar(vector_array_base self, double a)
    cdef void itruediv_vector(vector_array_base self, vector_array_base a)
    cdef vector_array_base negate(vector_array_base self)
    cdef vector_array_base copy(vector_array_base self)
    cdef vector_array_base absolute(vector_array_base self)

cdef vector_array_base vector_array_zeros_c(shape)

cdef vector_array_base vector_array_ones_c(shape)
cdef vector_array_base vector_array_c(param)
