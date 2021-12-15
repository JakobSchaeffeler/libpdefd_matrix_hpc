import cython
import numpy as np
cimport numpy as np
from ctypes import *
import sys
import os
import time
import array
import backend
sys.path.append(os.path.dirname(__file__))
import libpdefd_vector_array

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
    cdef dict __dict__
    cdef public double[:] data_as
    cdef public int c_length
    #cdef public int length

    def __init__(self, double[:] data = None, shape = None):
        self.data_as = data
        self.length = None

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
            if len(i) != len(self.shape):
                raise Exception("Trying to set item in array with wrong shape")
            # i = [l,n,m] (shape:x,y,z)==> [1,2,4] = l*(y*z) + n*z + m
            j = 0
            for k in range(len(i)):
                tmp = i[k]
                for l in range(k+1, len(i)):
                    tmp *= self.shape[l]
                j += tmp
            return self.data_as[j]

    def __setitem__(self, i, data):
        if isinstance(i, int):
            self.data_as[i] = data
        elif isinstance(i, tuple):
            if len(i) != len(self.shape):
                raise Exception("Trying to set item in array with wrong shape")
            # i = [l,n,m] (shape:x,y,z)==> [1,2,4] = l*(y*z) + n*z + m
            j = 0
            for k in range(len(i)):
                tmp = i[k]
                for l in range(k+1, len(i)):
                    tmp *= self.shape[l]
                j += tmp
            self.data_as[j] = data
        return


    def __add__(self, a):
        #cdef np.ndarray[float, ndim=1] res_c = np.empty(self.length)
        #cdef double *res = <double*> res_c
        #result = np.zeros((self.shape), dtype=np.cdouble)
        mkl_set_num_threads(backend.get_num_threads())

        cdef double[::1] res
        res = array.array("d", np.empty(self.length, dtype=np.double))
        cdef double* res_ptr = &res[0]
        cdef double[::1] arg1 = self.data_as
        cdef double[::1] arg2
        cdef double a_c

        if isinstance(a, self.__class__):
            arg2 = a.data_as
            start = time.time()
            vdAdd(self.c_length, &arg1[0], &arg2[0], res_ptr)
            end = time.time()
            print(end-start)
            return self.__class__(res, self.shape)
        if isinstance(a, float):
            a_c = a
            vdLinearFrac(self.c_length, &arg1[0],
                         &arg1[0], 1, a_c,
                             0, 1, res_ptr)
            return self.__class__(res, self.shape)
        raise Exception("Unsupported type '"+str(type(a))+"'")

    def to_numpy_array(self):
        """
        Return numpy array
        """
        print(self.data_as[0:3])
        return np.frombuffer(self.data_as, dtype=float)


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    retval = vector_array_base()
    if isinstance(param, np.ndarray):
        retval.data_as = array.array('d', param)
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
        retval.data_as = array.array(param)
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
    retval.data_as = array.array('d', [0]*retval.length)
    print(retval.data_as)
    print(retval[0,0])
    print(retval[0,1])
    print(retval[1,0])
    print(retval[1,1])
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
    retval.data_as = array.array('d', np.ones(shape))
    return retval


