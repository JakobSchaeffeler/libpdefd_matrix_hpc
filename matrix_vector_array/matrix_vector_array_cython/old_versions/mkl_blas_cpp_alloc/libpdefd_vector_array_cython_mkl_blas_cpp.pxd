import numpy as np
cimport numpy as np
cimport cvector

cdef class vector_array_base():
    cdef cvector.Vector _vector
    cdef public tuple shape
    cdef void dealloc(self)
    cdef vector_array_base add_vector(self, vector_array_base a)
    cdef void iadd_vector(self, vector_array_base a)
    cdef vector_array_base add_scalar(self, double a)
    cdef void iadd_scalar(self, double a)
    cdef vector_array_base mul_vector(self, vector_array_base a)
    cdef vector_array_base mul_scalar(self, double a)
    cdef void imul_vector(self, vector_array_base a)
    cdef void imul_scalar(self, double a)
    cdef vector_array_base sub_vector(self, vector_array_base a)
    cdef vector_array_base sub_scalar(self, double a)
    cdef void isub_vector(self, vector_array_base a)
    cdef void isub_scalar(self, double a)
    cdef vector_array_base rsub_vector(self, vector_array_base a)
    cdef vector_array_base rsub_scalar(self, double a)
    cdef vector_array_base pow_vector(self, vector_array_base a)
    cdef vector_array_base pow_scalar(self, double a)
    cdef vector_array_base truediv_vector(self, vector_array_base a)
    cdef vector_array_base truediv_scalar(self, double a)
    cdef vector_array_base rtruediv_vector(self, vector_array_base a)
    cdef vector_array_base rtruediv_scalar(self, double a)
    cdef void itruediv_vector(self, vector_array_base a)
    cdef void itruediv_scalar(self, double a)
    cdef np.ndarray convert_to_numpy_array(self)

cdef vector_array_base vector_array_c(param)



