import numpy as np
cimport numpy as np

cdef class vector_array_base():
    cdef public int length
    cdef double* data
    cdef vector_array_base add_vector(self, vector_array_base a)
    cdef iadd_vector(self, vector_array_base a)
    cdef vector_array_base add_scalar(self, double a)
    cdef iadd_scalar(self, double a)
    cdef np.ndarray convert_to_numpy_array(self)
