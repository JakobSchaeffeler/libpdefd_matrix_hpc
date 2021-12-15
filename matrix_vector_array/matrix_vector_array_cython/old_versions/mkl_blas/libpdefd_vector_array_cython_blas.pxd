import numpy as np
cimport numpy as np

cdef class vector_array_base():
    cdef public object data_as
    cdef public int c_length
    cdef public int length
    cdef public tuple shape
    cdef vector_array_base add_vector(self, vector_array_base a)
    cdef iadd_vector(self, vector_array_base a)
    cdef vector_array_base add_scalar(self, double a)
    cdef iadd_scalar(self, double a)

cdef vector_array_base vector_array_c(param)

