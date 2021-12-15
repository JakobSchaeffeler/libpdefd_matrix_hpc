import numpy as np
cimport numpy as np
np.import_array() # initialize C API to call PyArray_SimpleNewFromData
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


cdef extern from "mkl.h":
    cdef double* vdAdd (int length, double* arr1, double* arr2, double* res) nogil
    cdef double* vdLinearFrac (int length, double* arr1, double* arr2, double fac1,
                               double scal1, double fac2, double scal2, double* res) nogil
    cdef void mkl_set_num_threads(int nt) nogil
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
        self.data = NULL


    def __dealloc__(self):
        if self.data != NULL:
            mkl_free(self.data)
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
        cdef double* res_data = <double*> malloc(self.length*sizeof(double))
        if not res_data:
            raise MemoryError()
        vdAdd(self.length, self.data, a.data, res_data)
        cdef vector_array_base res = vector_array_base()
        res.data = res_data
        res.length = self.length
        return res

    cdef iadd_vector(self, vector_array_base a):
        vdAdd(self.length, self.data, a.data, self.data)
        return

    cdef vector_array_base add_scalar(self, double a):
        cdef double* res_data = <double*> malloc(self.length*sizeof(double))
        cdef vector_array_base res = vector_array_base()
        vdLinearFrac(self.length, self.data, self.data, 1, a, 0, 1, res_data)
        res.data = res_data
        res.length = self.length
        return res

    cdef iadd_scalar(self, double a):
        vdLinearFrac(self.length, self.data, self.data,1 , a, 0, 1, self.data)
        return

    def to_numpy_array(self):
        return vector_array_base.convert_to_numpy_array(self)

    cdef np.ndarray convert_to_numpy_array(self):
        cdef np.npy_intp dims = self.length
        cdef np.ndarray a = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_DOUBLE, <void*> self.data)
        return a


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    return cython_vector_array(param)


cdef cython_vector_array(np.ndarray param):
    cdef np.ndarray[double, ndim=1] param_flat
    cdef int length
    cdef double* res_data
    cdef vector_array_base res = vector_array_base()
    if isinstance(param, np.ndarray):
        res_data = <double*> malloc(param.size *sizeof(double))
        param_flat = param.reshape((param.size,))
        for i in range(param.size):
            res_data[i] = param_flat[i]
        length = <int> param.size
        res.length = length
        res.data = res_data
        return res
    raise Exception("Type '"+str(type(param))+"' of param not supported")

