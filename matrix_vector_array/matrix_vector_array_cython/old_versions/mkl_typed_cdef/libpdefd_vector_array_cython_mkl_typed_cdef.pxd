import numpy as np
cimport numpy as np

cdef class vector_array_base():
    cdef dict __dict__
    cdef public object data_as
    cdef public int length
    cdef public tuple shape
    cdef vector_array_base add_vector(self, vector_array_base a)
    cdef vector_array_base iadd_vector(self, vector_array_base a)
    cdef vector_array_base add_scalar(self, double a)
    cdef vector_array_base iadd_scalar(self, double a)
