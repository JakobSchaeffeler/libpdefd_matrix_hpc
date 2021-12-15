import cython
import numpy as np
cimport numpy as np
from ctypes import *
import sys
import os
import time
import array
sys.path.append(os.path.dirname(__file__))
#import libpdefd_vector_array


cdef extern from "mkl.h":
    cdef double* vdAdd (int length, double* arr1, double* arr2, double* res) nogil
    cdef double* vdLinearFrac (int length, double* arr1, double* arr2, double fac1,
                               double scal1, double fac2, double scal2, double* res) nogil
    cdef void mkl_set_num_threads(int nt) nogil
    cdef void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy) nogil
    cdef void cblas_daxpy(const int n, const double a, const double *x, const int incx, double *y, const int incy) nogil


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

    """
    def __init__(self, np.ndarray data = None, shape = None):
        self.data_as = data
        self.length = 0
        self.c_length = 0
        self.shape = None
    

        if self.data_as is not None:
            self.shape = shape
            self.length = np.prod(shape)
            self.c_length = <int> self.length
    """

    def __getitem__(self, i):
        cdef int j
        if isinstance(i, int):
            j = i
            return self.data_as[j]
        elif isinstance(i, tuple):
            return self.data_as[j]

    def __setitem__(self, i, data):
        if isinstance(i, int):
            self.data_as[i] = data
        elif isinstance(i, tuple):
            self.data_as[i] = data
        return

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



    cdef vector_array_base add_scalar(self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double a_c = a
        vdLinearFrac(self.c_length, self_ptr,
                         self_ptr, 1, a_c,
                         0, 1, res_ptr)
        return self.__class__(res, self.shape)



    cdef vector_array_base add_vector(self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef np.ndarray[double, ndim=1] arg2 = a.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double* a_ptr = &arg2[0]

        cblas_dcopy(self.c_length, a_ptr, 1, res_ptr, 1)
        cblas_daxpy(self.c_length, 1.0, self_ptr, 1, res_ptr, 1)
        return self.__class__(res, self.shape)


    def __iadd__(self, a):
        cdef double a_c
        cdef vector_array_base vec_b
        if isinstance(a, vector_array_base):
            vec_b = a
            vector_array_base.iadd_vector(self,vec_b)
            return self
        elif isinstance(a, float):
            a_c = a
            vector_array_base.iadd_scalar(self, a_c)
            return self
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")



    cdef iadd_scalar(self, double a):
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double a_c = a
        vdLinearFrac(self.c_length, self_ptr,
                     self_ptr, 1, a_c,
                     0, 1, self_ptr)
        return



    cdef iadd_vector(self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef np.ndarray[double, ndim=1] arg2 = a.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double* a_ptr = &arg2[0]
        cblas_daxpy(self.c_length, 1.0, a_ptr, 1, self_ptr, 1)
        return


    def to_numpy_array(self):
        """
        Return numpy array
        """
        return self.data_as

def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    return vector_array_c(param)

cdef vector_array_base vector_array_c(param):
    retval = vector_array_base()
    cdef np.ndarray[double, ndim=1] param_flat
    cdef np.ndarray[double, ndim=1] res_flat
    cdef double* param_ptr
    cdef double* res
    if isinstance(param, np.ndarray):
        res_flat = np.empty(np.prod(param.shape))
        res = &res_flat[0]
        param_flat = param.reshape((param.size,))
        param_ptr = &param_flat[0]
        cblas_dcopy(np.prod(param.shape), param_ptr, 1, res, 1)
        retval.data_as = res_flat
        retval.shape = param.shape
        retval.length = np.prod(param.shape)
        retval.c_length = <int> retval.length
        return retval

    if isinstance(param, vector_array_base):
        retval.data_as = param.data_as
        retval.shape = param.shape
        retval.length = param.length
        retval.c_length = <int> retval.length
        return retval

    if isinstance(param, list):
        retval.data_as = np.array(param)
        retval.shape = (len(param),)
        retval.length = len(param)
        retval.c_length = <int> retval.length

        return retval

    raise Exception("Type '"+str(type(param))+"' of param not supported")


def vector_array_zeros(shape, dtype=None):
    """
    Return array of shape with zeros
    """
    print(shape)
    retval = vector_array_base()
    retval.shape = shape
    retval.length = np.prod(shape)
    retval.c_length = <int> retval.length
    retval.data_as = np.zeros(shape)
    return retval



def vector_array_zeros_like(data, dtype=None):
    """
    Return zero array of same shape as data
    """
    return vector_array_zeros(data.shape, dtype=dtype)


def vector_array_ones(shape, dtype=None):
    """
    Return array of shape with ones
    """
    retval = vector_array_base()
    retval.shape = shape
    retval.length = np.prod(shape)
    retval.c_length = <int> retval.length
    retval.data_as = np.ones(shape)
    return retval


def test_iadd():
    print("starting setup")
    cdef np.ndarray[double, ndim=1] x = np.random.rand(16000000)
    print("converting...")
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(16000000)
    cdef double *y_ptr= &y[0]
    cdef np.ndarray[double, ndim=1] a = np.empty(16000000)
    cdef np.ndarray[double, ndim=1] b = np.empty(16000000)
    cdef double* param1 = &a[0]
    cdef double* param2 = &b[0]
    print("setup finished")
    for i in range(1,24):
        mkl_set_num_threads(<int> i)
        x = np.random.rand(16000000)
        y = np.random.rand(16000000)
        x_ptr= &x[0]
        y_ptr= &y[0]
        a = np.empty(16000000)
        b = np.empty(16000000)
        param1 = &a[0]
        param2 = &b[0]
        cblas_dcopy(16000000, x_ptr, 1, param1, 1)
        cblas_dcopy(16000000, y_ptr, 1, param2, 1)
        print("copy finished")
        for _ in range(100):
            cblas_daxpy(16000000, 1.0, param2, 1, param1, 1)




