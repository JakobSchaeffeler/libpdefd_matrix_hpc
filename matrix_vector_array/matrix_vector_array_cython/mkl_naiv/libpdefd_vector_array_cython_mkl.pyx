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


# cdef extern from "mkl.h":
#     cdef void vdAdd (int length, double* arr1, double* arr2, double* res) nogil
#     cdef void vdSub (int length, double* arr1, double* arr2, double* res) nogil
#     cdef void vdMul (int length, double* arr1, double* arr2, double* res) nogil
#     cdef void vdPow (int length, double* arr1, double* arr2, double* res) nogil
#     cdef void vdPowx (int length, double* arr1, double scal, double* res) nogil
#     cdef void vdDiv (int length, double* arr1, double* arr2, double* res) nogil
#     cdef void vdAbs (const int n, const double* a, double* y) nogil
#
#     cdef void cblas_dcopy(int length, double* arr1, int incx, double* res, int incy) nogil
#     cdef void cblas_dscal(int length, double a, double* arr, int step) nogil
#     cdef int cblas_idamax (const int n, const double *x, const int incx) nogil
#     cdef int cblas_idamin (const int n, const double *x, const int incx) nogil
#
#     cdef double* vdLinearFrac (int length, double* arr1, double* arr2, double fac1,
#                                double scal1, double fac2, double scal2, double* res) nogil
#     cdef void mkl_set_num_threads(int nt) nogil


mkl = cdll.LoadLibrary("libmkl_rt.so")



"""
The vector-array class is one which
 * stores multi-dimensional array data representing spatial information and
 * allows efficient matrix-vector multiplications with the compute matrix class
"""

class vector_array_base:
    """
    Array container to store varying data during simulation

    This data is stored on a regular high(er) dimensional Cartesian grid.
    """

    def __init__(self, data = None, shape = None):
        self.data_as = None

        if data is not None:
            if isinstance(data, self.__class__):
                self.data_as = data.data_as
            elif isinstance(data, np.ndarray):
                self.data_as = data.ctypes.data_as(POINTER(c_double))
                shape = data.shape
            else:
                self.data_as = data
        self.shape = None
        self.length = None
        self.c_length = None

        if self.data_as is not None:
            self.shape = shape
            self.length = np.prod(shape)
            self.c_length = c_int(self.length)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.data_as[i]
        elif isinstance(i, tuple):
            if len(i) != len(self.shape):
                raise Exception("Trying to get item in array with wrong shape")
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
        if isinstance(a, self.__class__):
            res_np = np.empty(self.length)
            res = res_np.ctypes.data_as(POINTER(c_double))
            mkl.vdAdd(self.c_length, self.data_as, a.data_as, res)
            return self.__class__(res, self.shape)

        if isinstance(a, float):
            res_np = np.empty(self.length)
            res = res_np.ctypes.data_as(POINTER(c_double))

            mkl.vdLinearFrac(self.c_length, self.data_as,
                             self.data_as, c_double(1), c_double(a),
                             c_double(0), c_double(1), res)
            return self.__class__(res, self.shape)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    def __iadd__(self, a):
        if isinstance(a, self.__class__):
            mkl.vdAdd(self.c_length, self.data_as, a.data_as, self.data_as)
        elif isinstance(a, float):
            mkl.vdLinearFrac(self.c_length, self.data_as,
                             self.data_as, c_double(1), c_double(a),
                             c_double(0), c_double(1), self.data_as)
        else:
            raise Exception("Unsupported type '"+str(type(a))+"'")
        return self

    def __radd__(self, a):
        if isinstance(a, self.__class__):
            res_np = np.empty(self.length)
            res = res_np.ctypes.data_as(POINTER(c_double))
            mkl.vdAdd(self.c_length, self.data_as, a.data_as, res)
            return self.__class__(res, self.shape)
        if isinstance(a, float):
            res_np = np.empty(self.length)
            res = res_np.ctypes.data_as(POINTER(c_double))

            mkl.vdLinearFrac(self.c_length, self.data_as,
                             self.data_as, c_double(1), c_double(a),
                             c_double(0), c_double(1), res)
            return self.__class__(res, self.shape)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    def __sub__(self, a):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        if isinstance(a, self.__class__):
            mkl.vdSub(self.c_length, self.data_as, a.data_as, res)
            return self.__class__(res, self.shape)

        if isinstance(a, float):
            mkl.vdLinearFrac(self.c_length, self.data_as,
                             self.data_as, c_double(1), c_double(-a),
                             c_double(0), c_double(1), res)
            return self.__class__(res, self.shape)



    def __isub__(self, a):
        if isinstance(a, self.__class__):
            mkl.vdSub(self.c_length, self.data_as, a.data_as, self.data_as)
        else:
            mkl.vdLinearFrac(self.c_length, self.data_as,
                             self.data_as, c_double(1), c_double(-a),
                             c_double(0), c_double(1), self.data_as)
        return self

    def __rsub__(self, a):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        if isinstance(a, self.__class__):
            mkl.vdSub(self.c_length, a.data_as, self.data_as, res)
            return self.__class__(res, self.shape)
        else:
            mkl.vdLinearFrac(self.c_length, self.data_as,
                             self.data_as, c_double(-1), c_double(a),
                             c_double(0), c_double(1), res)
            return self.__class__(res, self.shape)



    def __mul__(self, a):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))

        if isinstance(a, self.__class__):
            mkl.vdMul(self.c_length, self.data_as,
                      a.data_as, res)
            return self.__class__(res, self.shape)
        else:
            mkl.cblas_daxpby(self.c_length, c_double(a), self.data_as, 1, c_double(0.0), res, 1)
            return self.__class__(res, self.shape)

    def __imul__(self, a):
        if isinstance(a, self.__class__):
            mkl.vdMul(self.c_length, self.data_as, a.data_as, self.data_as)
        else:
            mkl.cblas_dscal(self.c_length, c_double(a), self.data_as, 1)
        return self

    def __rmul__(self, a):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))

        if isinstance(a, self.__class__):
            mkl.vdMul(self.c_length, self.data_as,
                      a.data_as, res)
            return self.__class__(res, self.shape)
        else:
            mkl.cblas_daxpby(self.c_length, c_double(a), self.data_as, 1, c_double(0.0), res, 1)
            return self.__class__(res, self.shape)

    def __pow__(self, a):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        if isinstance(a, self.__class__):
            mkl.vdPow(self.c_length, self.data_as,
                      a.data_as, res)
            return self.__class__(res, self.shape)
        else:
            mkl.vdPowx(self.c_length, self.data_as, c_double(a),
                       res)
        return self.__class__(res, self.shape)

    def __truediv__(self, a):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        if isinstance(a, self.__class__):
            mkl.vdDiv(self.c_length, self.data_as, a.data_as, res)
            return self.__class__(res, self.shape)
        else:
            mkl.cblas_daxpby(self.c_length, c_double(1/a), self.data_as, 1, c_double(0.0), res, 1)
            return self.__class__(res, self.shape)

    def __rtruediv__(self, a):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        if isinstance(a, self.__class__):
            mkl.vdDiv(self.c_length, a.data_as, self.data_as,
                      res)
            return self.__class__(res, self.shape)
        else:
            mkl.vdInv(self.c_length, self.data_as, res)
            mkl.cblas_dscal(self.c_length, c_double(a), res, 1)

            return self.__class__(res, self.shape)

    def __itruediv__(self, a):
        if isinstance(a, self.__class__):
            mkl.vdDiv(self.c_length, self.data_as,
                      a.data_as, self.data_as)
            return self
        else:
            mkl.cblas_daxpby(self.c_length, c_double(1/a), self.data_as, 1, c_double(0.0), self.data_as, 1)
            return self

    def __neg__(self):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        mkl.vdLinearFrac(self.c_length, self.data_as,
                         self.data_as, c_double(-1), c_double(0),
                         c_double(0), c_double(1), res)
        return self.__class__(res, self.shape)

    def __pos__(self):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        mkl.cblas_dcopy(self.c_length, self.data_as, 1, res, 1)
        return self.__class__(res, self.shape)


    def abs(self):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        mkl.vdAbs(self.c_length, self.data_as, res)
        return self.__class__(res, self.shape)

    def reduce_min(self):
        ArrayType = c_double*self.length
        return min(np.frombuffer(ArrayType.from_address(addressof(self.data_as.contents))))


    def reduce_minabs(self):
        pos = mkl.cblas_idamin(self.c_length, self.data_as, 1)
        return abs((self.data_as)[pos])

    def reduce_max(self):
        ArrayType = c_double*self.length
        return max(np.frombuffer(ArrayType.from_address(addressof(self.data_as.contents))))

    def reduce_maxabs(self):
        pos = mkl.cblas_idamax(self.c_length, self.data_as, 1)
        return abs((self.data_as)[pos])


    def copy(self):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        mkl.cblas_dcopy(self.c_length, self.data_as, 1, res, 1)
        return self.__class__(res, self.shape)


    def kron_vector(self, data):
        """
        We need a Kronecker product to assemble the RHS of the FD operators once extending it to higher dimensions.
        """
        assert isinstance(data, vector_array_base)
        assert len(data.shape) == 1
        assert len(self.shape) == 1
        d = np.kron(self.to_numpy_array(), data.to_numpy_array())
        return self.__class__(d)


    def set_all(self, scalar_value):
        # data[i] = (0 * data[i] + scalar_value) /(0 * data[i] + 1)
        mkl.vdLinearFrac(self.c_length, self.data_as,
                         self.data_as, c_double(0), c_double(scalar_value),
                         c_double(0), c_double(1), self.data_as)
        return


    def flatten(self):
        return self.__class__(self.data_as, shape=np.prod(self.shape))

    def num_elements(self):
        return self.length

    def __str__(self):
        retstr = "PYSMarray: "
        retstr += str(self.shape)
        return retstr

    def to_numpy_array(self):
        """
        Return numpy array
        """
        ArrayType = c_double*self.length
        return np.frombuffer(ArrayType.from_address(addressof(self.data_as.contents))).reshape(self.shape)


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    retval = vector_array_base()
    if isinstance(param, np.ndarray):
        param_cpy = param.copy()
        retval.data_as = param_cpy.ctypes.data_as(POINTER(c_double))
        retval.shape = param.shape
        retval.length = np.prod(param.shape)
        retval.c_length = c_int(retval.length)
        return retval

    if isinstance(param, vector_array_base):
        retval.data_as = param.data_as.copy()
        retval.shape = param.shape
        retval.length = param.length
        retval.c_length = c_int(retval.length)
        return retval

    if isinstance(param, list):
        retval.data_as = np.array(param, dtype=dtype).ctypes.data_as(POINTER(c_double))
        retval.shape = (len(param),)
        retval.length = len(param)
        retval.c_length = c_int(retval.length)

        return retval
    if isinstance(param, POINTER(c_double)):
        retval.data_as = param
        retval.shape = shape
        retval.length = np.prod(shape)
        retval.c_length = c_int(retval.length)

    raise Exception("Type '"+str(type(param))+"' of param not supported")


def vector_array_zeros(shape, dtype=None):
    """
    Return array of shape with zeros
    """
    retval = vector_array_base()
    retval.shape = shape
    retval.length = np.prod(shape)
    retval.c_length = c_int(retval.length)
    retval.data_as = np.zeros(shape).ctypes.data_as(POINTER(c_double))
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
    _data = np.ones(shape)
    retval.data_as = _data.ctypes.data_as(POINTER(c_double))
    retval.shape = shape
    retval.length = np.prod(shape)
    retval.c_length = c_int(retval.length)
    return retval


