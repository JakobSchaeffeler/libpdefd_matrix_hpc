# distutils: sources = vector.c
# distutils: include_dirs = .

cimport cvector
import numpy as np
cimport numpy as np
np.import_array() # initialize C API to call PyArray_SimpleNewFromData
from libc.stdlib cimport malloc, free

"""
The vector-array class is one which
 * stores multi-dimensional array data representing spatial information and
 * allows efficient matrix-vector multiplications with the compute matrix class
"""

cdef class vector_array_base:
    """
    Array container to store varying data during simulation

    This data is stored on a regular high(er) dimensional Cartesian grid.
    """
    cdef cvector.Vector _vector
    cdef public int length
    cdef double* data

    def __cinit__(self):
        self.length = 0

    def __dealloc__(self):
        if self.data != NULL:
            cvector.vector_free(self._vector)
            self.data = NULL

    def __add__(self, a):
        cdef double a_c
        cdef vector_array_base vec_b
        if isinstance(a, vector_array_base):
            vec_b = a
            return vector_array_base.add_vec(self,vec_b)
        elif isinstance(a, float):
            a_c = a
            return vector_array_base.add_scal(self, a_c)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    def __iadd__(self, a):
        if isinstance(a, vector_array_base):
            return vector_array_base.iadd_vec(self, a)
        elif isinstance(a, float):
            return vector_array_base.iadd_scal(self, a)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef vector_array_base add_vec(self, vector_array_base a):
        cdef double* res_data = cvector.add_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        return res

    cdef vector_array_base iadd_vec(self, vector_array_base a):
        cvector.iadd_vector(self._vector, a._vector)
        return self

    cdef vector_array_base add_scal(self, double a):
        cdef double* res_data = cvector.add_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        return res

    cdef vector_array_base iadd_scal(self, double a):
        cvector.iadd_scalar(self._vector, a)
        return self

    def __mul__(self, a):
        cdef double a_c
        cdef vector_array_base vec_b
        if isinstance(a, vector_array_base):
            vec_b = a
            return vector_array_base.mul_vec(self,vec_b)
        elif isinstance(a, float):
            a_c = a
            return vector_array_base.mul_scal(self, a_c)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef vector_array_base mul_vec(self, vector_array_base a):
        cdef double* res_data = <double*> malloc(self.length*sizeof(double))
        cvector.mul_vector(self._vector, a._vector, res_data)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        return res

    cdef vector_array_base mul_scal(self, double a):
        cdef double* res_data = <double*> malloc(self.length*sizeof(double))
        cvector.mul_scalar(self._vector, a, res_data)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        return res

    def __imul__(self, a):
        if isinstance(a, vector_array_base):
            vector_array_base.imul_vec(self, a)
            return self
        elif isinstance(a, float):
            vector_array_base.imul_scal(self, a)
            return self
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef vector_array_base imul_vec(self, vector_array_base a):
        cvector.imul_vector(self._vector, a._vector)
        return

    cdef vector_array_base imul_scal(self, double a):
        cvector.imul_scalar(self._vector, a)
        return


    def get_elem(self):
        cdef double d = cvector.get_data(self._vector)[0]
        return d

    def to_numpy_array(self):
        return vector_array_base.convert_to_numpy_array(self)

    cdef np.ndarray convert_to_numpy_array(self):
        cdef np.npy_intp dims = cvector.get_length(self._vector)
        cdef np.ndarray a = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_DOUBLE, <void*> cvector.get_data(self._vector))
        return a


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    cdef np.ndarray[double, ndim=1] param_flat
    cdef double* res_data
    cdef int length
    cdef vector_array_base res = vector_array_base()
    cdef cvector.Vector vec
    if isinstance(param, np.ndarray):
        res_data = cvector.malloc_vector(param.size *sizeof(double),512)
        param_flat = param.reshape((param.size,))
        for i in range(param.size):
            res_data[i] = param_flat[i]
        length = <int> param.size
        res.length = length
        res.data = res_data
        vec = cvector.vector_new(res_data, length)
        res._vector = vec
        return res
    raise Exception("Type '"+str(type(param))+"' of param not supported")

