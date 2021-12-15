import numpy as np
cimport numpy as np
cimport cvector

cdef class vector_array_base():
    cdef cvector.Vector _vector
    cdef public int length
    cdef double* data
    cdef tuple shape
    cdef vector_array_base add_vector(self, vector_array_base a)
    cdef iadd_vector(self, vector_array_base a)
    cdef vector_array_base add_scalar(self, double a)
    cdef iadd_scalar(self, double a)
    cdef vector_array_base mul_vector(self, vector_array_base a)
    cdef vector_array_base mul_scalar(self, double a)
    cdef vector_array_base imul_vector(self, vector_array_base a)
    cdef vector_array_base imul_scalar(self, double a)
    cdef np.ndarray convert_to_numpy_array(self)



cdef vector_array_base add_scalar(vector_array_base x, double a)
cdef vector_array_base add_vector(vector_array_base x, vector_array_base a)
