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
from libc.stdlib cimport malloc, free


cdef extern from "mkl.h":
    cdef double* vdAdd (int length, double* arr1, double* arr2, double* res) nogil
    cdef double* vdLinearFrac (int length, double* arr1, double* arr2, double fac1,
                               double scal1, double fac2, double scal2, double* res) nogil
    cdef void mkl_set_num_threads(int nt) nogil


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


    def __init__(self, np.ndarray data = None, shape = None):
        self.data_as = data
        #self.length = None

        self.shape = None

        if self.data_as is not None:
            self.shape = shape
            self.length = <int> np.prod(shape)

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


    cdef vector_array_base add_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double a_c = a
        vdLinearFrac(self.length, self_ptr,
                         self_ptr, 1, a_c,
                         0, 1, res_ptr)
        return self.__class__(res, self.shape)



    cdef vector_array_base add_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef np.ndarray[double, ndim=1] arg2 = a.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double* a_ptr = &arg2[0]
        vdAdd(self.length, self_ptr, a_ptr, res_ptr)
        return self.__class__(res, self.shape)

    def __add__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.add_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.add_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    def __iadd__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.iadd_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.iadd_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef vector_array_base iadd_vector(self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef np.ndarray[double, ndim=1] arg2 = a.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double* a_ptr = &arg2[0]
        vdAdd(self.length, self_ptr, a_ptr, self_ptr)
        return self

    cdef vector_array_base iadd_scalar(self, double a):
        cdef np.ndarray[double, ndim=1] arg1 = self.data_as
        cdef double* self_ptr = &arg1[0]
        cdef double a_c = a
        vdLinearFrac(self.length, self_ptr,
                     self_ptr, 1, a_c,
                     0, 1, self_ptr)
        return self




    def to_numpy_array(self):
        """
        Return numpy array
        """
        return self.data_as


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    retval = vector_array_base()
    if isinstance(param, np.ndarray):
        retval.data_as = param.copy().reshape((param.size,))
        retval.shape = param.shape
        retval.length = <int> np.prod(param.shape)
        return retval

    if isinstance(param, vector_array_base):
        retval.data_as = param.data_as.copy()
        retval.shape = param.shape
        retval.length = <int> np.prod(param.shape)
        return retval

    if isinstance(param, list):
        retval.data_as = np.array(param)
        retval.shape = (len(param),)
        retval.length = <int> len(param)

        return retval

    raise Exception("Type '"+str(type(param))+"' of param not supported")


def vector_array_zeros(shape, dtype=None):
    """
    Return array of shape with zeros
    """
    retval = vector_array_base()
    retval.shape = shape
    retval.length = <int> np.prod(shape)
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
    retval.length = <int> np.prod(shape)
    retval.data_as = np.ones(shape)
    return retval


