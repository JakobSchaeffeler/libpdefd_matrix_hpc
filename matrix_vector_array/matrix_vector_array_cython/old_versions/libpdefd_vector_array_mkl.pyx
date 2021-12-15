import cython
import numpy as np
cimport numpy as np
from ctypes import *
from cython.parallel cimport prange
import sys
import os
import time
import array
from cpython cimport array
cimport openmp
sys.path.append(os.path.dirname(__file__))
import libpdefd_vector_array
import backend


cdef extern from "mkl.h":
    cdef double* vdAdd (int length, double* arr1, double* arr2, double* res) nogil
    cdef double* vdLinearFrac (int length, double* arr1, double* arr2, double fac1,
                               double scal1, double fac2, double scal2, double* res) nogil
    cdef int mkl_get_max_threads() nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef iadd_omp(double[:] x, double[:] y):
        cdef Py_ssize_t i
        cdef int num_threads = backend.get_num_threads()
        for i in prange(x.shape[0], nogil=True, num_threads = num_threads):
            x[i] += y[i]
        return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef iadd_scalar_omp(double[:] x, double y):
        cdef Py_ssize_t i
        cdef int num_threads = backend.get_num_threads()
        for i in prange(x.shape[0], nogil=True, num_threads = num_threads):
            x[i] += y
        return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef add_omp(double[:] x, double[:] y, double[:] z):
        cdef Py_ssize_t i
        cdef int num_threads = backend.get_num_threads()
        for i in prange(x.shape[0], nogil=True, num_threads = num_threads):
            z[i] = y[i] + x[i]
        return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef add_scalar_omp(double[:] x, double y, double[:] z):
        cdef Py_ssize_t i
        cdef int num_threads = backend.get_num_threads()
        for i in prange(x.shape[0], nogil=True, num_threads= num_threads):
            z[i] = x[i] + y
        return


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
    cdef dict __dict__
    cdef public object data_as
    cdef public int c_length
    cdef public int length

    def __init__(self, np.ndarray data = None, shape = None):
        self.data_as = data
        #self.data_pointer = data
        #self.length = None

        self.shape = None

        if self.data_as is not None:
            self.shape = shape
            self.length = np.prod(shape)
            self.c_length = <int> self.length

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def __add__(self, a):
        #cdef np.ndarray[float, ndim=1] res_c = np.empty(self.length)
        #cdef double *res = <double*> res_c
        #result = np.zeros((self.shape), dtype=np.cdouble)
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length, dtype=np.double)
        cdef double a_c

        if isinstance(a, self.__class__):
            add_omp(self.data_as, a.data_as, res)
            return self.__class__(res, self.shape)
        if isinstance(a, float):
            a_c = a
            add_scalar_omp(self.data_as, a_c, res)
            return self.__class__(res, self.shape)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def __iadd__(self, a):
        cdef double a_c
        cdef array.array x
        if isinstance(a, self.__class__):
            iadd_omp(self.data_as, a.data_as)
            return
        if isinstance(a, float):
            a_c = a
            iadd_scalar_omp(self.data_as, a_c)
            return

        raise Exception("Unsupported type '"+str(type(a))+"'")

    def to_numpy_array(self):
        """
        Return numpy array
        """
        return self.data_as

    def get_num_threads(self):
        print(mkl_get_max_threads())
        return mkl_get_max_threads()




def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    retval = vector_array_base()
    if isinstance(param, np.ndarray):
        retval.data_as = param
        retval.shape = param.shape
        retval.length = np.prod(param.shape)
        retval.c_length = <int> retval.length
        retval.data_pointer = array.array('d', retval.data_as)

        return retval

    if isinstance(param, vector_array_base):
        retval.data_as = param.data_as
        retval.shape = param.shape
        retval.length = param.length
        retval.c_length = <int> retval.length
        retval.data_pointer = array.array('d', retval.data_as)

        return retval

    if isinstance(param, list):
        retval.data_as = np.array(param)
        retval.shape = (len(param),)
        retval.length = len(param)
        retval.c_length = <int> retval.length
        retval.data_pointer = array.array('d', retval.data_as)

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
    retval.data_pointer = array.array('d', retval.data_as)

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
    retval.data_pointer = array.array('d', retval.data_as)

    return retval


