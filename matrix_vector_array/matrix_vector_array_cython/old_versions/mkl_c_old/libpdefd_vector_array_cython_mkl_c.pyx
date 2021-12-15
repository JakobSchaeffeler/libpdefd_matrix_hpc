# distutils: sources = vector.c
# distutils: include_dirs = .

cimport cvector
import numpy as np
cimport numpy as np
np.import_array() # initialize C API to call PyArray_SimpleNewFromData
from libc.stdlib cimport malloc, free

cdef extern from "mkl.h":
    cdef void* mkl_malloc(size_t len, int n) nogil
    cdef void* mkl_free(void* ptr) nogil
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
            return vector_array_base.add_vector(self,vec_b)
        elif isinstance(a, float):
            a_c = a
            return vector_array_base.add_scalar(self, a_c)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")


    """
    add commutative ==> use add again
    """

    def __radd__(self, a):
        return self.__add__(a)


    def __iadd__(self, a):
        if isinstance(a, vector_array_base):
            vector_array_base.iadd_vector(self, a)
            return self
        elif isinstance(a, float):
            vector_array_base.iadd_scalar(self, a)
            return self
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef vector_array_base add_vector(self, vector_array_base a):
        cdef double* res_data = cvector.add_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        res.shape = self.shape
        return res

    cdef iadd_vector(self, vector_array_base a):
        cvector.iadd_vector(self._vector, a._vector)
        return

    cdef vector_array_base add_scalar(self, double a):
        cdef double* res_data = cvector.add_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        res.shape = self.shape
        return res

    cdef iadd_scalar(self, double a):
        cvector.iadd_scalar(self._vector, a)
        return

    def __mul__(self, a):
        cdef double a_c
        cdef vector_array_base vec_b
        if isinstance(a, vector_array_base):
            vec_b = a
            return vector_array_base.mul_vector(self,vec_b)
        elif isinstance(a, float):
            a_c = a
            return vector_array_base.mul_scalar(self, a_c)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    def __rmul__(self, a):
        cdef double a_c
        cdef vector_array_base vec_b
        if isinstance(a, vector_array_base):
            vec_b = a
            return vector_array_base.mul_vector(self,vec_b)
        elif isinstance(a, float):
            a_c = a
            return vector_array_base.mul_scalar(self, a_c)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")


    cdef vector_array_base mul_vector(self, vector_array_base a):
        cdef double* res_data = cvector.mul_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        res.shape = self.shape
        return res




    cdef vector_array_base mul_scalar(self, double a):
        cdef double* res_data = cvector.mul_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        vec = cvector.vector_new(res_data, self.length)
        res._vector = vec
        res.data = res_data
        res.shape = self.shape
        return res

    def __imul__(self, a):
        if isinstance(a, vector_array_base):
            vector_array_base.imul_vector(self, a)
            return self
        elif isinstance(a, float):
            vector_array_base.imul_scalar(self, a)
            return self
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")


    cdef vector_array_base imul_vector(self, vector_array_base a):
        cvector.imul_vector(self._vector, a._vector)
        return

    cdef vector_array_base imul_scalar(self, double a):
        cvector.imul_scalar(self._vector, a)
        return


    def copy(self):
        cdef vector_array_base res = vector_array_base()
        cdef double* res_data = <double*> malloc(self.length*sizeof(double))
        cvector.set_data(res._vector, res_data)
        cvector.copy(self._vector, res._vector)
        res.data = res_data
        res.length = self.length
        res.shape = self.shape
        return res


    def get_elem(self):
        cdef double d = cvector.get_data(self._vector)[0]
        return d

    def to_numpy_array(self):
        return vector_array_base.convert_to_numpy_array(self)

    cdef np.ndarray convert_to_numpy_array(self):
        cdef np.npy_intp dims = cvector.get_length(self._vector)
        cdef np.ndarray a = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_DOUBLE, <void*> cvector.get_data(self._vector))
        return a.reshape(self.shape)


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
        vec = cvector.vec_from_ptr(length, res_data)
        cvector.ptr_free(res_data)
        res.length = length
        res.data = cvector.get_data(vec)
        res.shape = param.shape
        res._vector = vec
        return res
    raise Exception("Type '"+str(type(param))+"' of param not supported")

cdef vector_array_base add_scalar(vector_array_base x, double a):
    cdef double* res_data = cvector.add_scalar(x._vector, a)
    cdef vector_array_base res = vector_array_base()
    vec = cvector.vector_new(res_data, x.length)
    res._vector = vec
    res.data = res_data
    res.shape = x.shape
    return res

cdef vector_array_base add_vector(vector_array_base x, vector_array_base a):
    cdef double* res_data = cvector.add_vector(x._vector, a._vector)
    cdef vector_array_base res = vector_array_base()
    vec = cvector.vector_new(res_data, x.length)
    res._vector = vec
    res.data = res_data
    res.shape = x.shape
    return res
