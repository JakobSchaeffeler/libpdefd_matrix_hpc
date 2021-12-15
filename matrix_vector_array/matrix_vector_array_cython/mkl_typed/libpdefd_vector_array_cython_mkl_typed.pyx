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
    cdef void vdAdd (int length, double* arr1, double* arr2, double* res) nogil
    cdef void vdMul (int length, double* arr1, double* arr2, double* res) nogil
    cdef void vdSub (int length, double* arr1, double* arr2, double* res) nogil
    cdef void vdDiv (int length, double* arr1, double* arr2, double* res) nogil
    cdef void vdAbs (int length, double* arr1, double* res) nogil
    cdef void vdPow (int length, double* arr1, double* arr2, double* res) nogil
    cdef void vdPowx (int length, double* arr1, double scal, double* res) nogil
    cdef void cblas_dcopy(int length, double* arr1, int incx, double* res, int incy) nogil
    cdef void cblas_dscal(int length, double a, double* arr, int step) nogil
    cdef int cblas_idamax (const int n, const double *x, const int incx) nogil
    cdef int cblas_idamin (const int n, const double *x, const int incx) nogil
    cdef void cblas_dcopy(int length, double* x, int incx, double* y, int incy) nogil
    cdef void cblas_daxpy(int length, double a, double* x, int incx, double* y, int incy) nogil
    cdef void cblas_daxpby(int length, double a, double* x, int incx, double b, double* y, int incy) nogil

    cdef void vdLinearFrac (int length, double* arr1, double* arr2, double scal1,
                               double shift1, double scal2, double shift2, double* res) nogil


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
        self.shape = ()
        self.length = 0

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
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.add_vector(self, a_vec)

        elif isinstance(a, float):
            a_scal = a
            return vector_array_base.add_scalar(self, a_scal)
        else:
            raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef vector_array_base add_vector(vector_array_base self, vector_array_base b):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdAdd(self.length, self.data_as, b.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    cdef vector_array_base add_scalar(vector_array_base self, double b):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdLinearFrac(self.length, self.data_as,
                     self.data_as, 1, b,
                     0, 1, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def __radd__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.add_vector(self,a_vec)

        elif isinstance(a, float):
            a_scal = a
            return vector_array_base.add_scalar(self,a_scal)
        else:
            raise Exception("Unsupported type '"+str(type(a))+"'")

    def __iadd__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            vector_array_base.iadd_vector(self, a_vec)
        elif isinstance(a, float):
            a_scal = a
            vector_array_base.iadd_scalar(self,a_scal)
        else:
            raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef void iadd_vector(vector_array_base self, vector_array_base b):
        vdAdd(self.length, self.data_as, b.data_as, self.data_as)
        return

    cdef void iadd_scalar(vector_array_base self, double b):
        vdLinearFrac(self.length, self.data_as,
                     self.data_as, 1, b,
                     0, 1, self.data_as)
        return


    cdef vector_array_base sub_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        vdLinearFrac(self.length, self.data_as,
                     self.data_as, 1, -a_c,
                     0, 1, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec


    cdef vector_array_base sub_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdSub(self.length, self.data_as, a.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def __sub__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.sub_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.sub_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef void isub_scalar(vector_array_base self, double a):
        vdLinearFrac(self.length, self.data_as,
                     self.data_as, 1, -a,
                     0, 1, self.data_as)
        return


    cdef void isub_vector(vector_array_base self, vector_array_base a):
        vdSub(self.length, self.data_as, a.data_as, self.data_as)
        return

    def __isub__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            vector_array_base.isub_vector(self, a_vec)
            return self
        if isinstance(a, float):
            a_scal = a
            vector_array_base.isub_scalar(self, a_scal)
            return self
        raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef vector_array_base rsub_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        vdLinearFrac(self.length, self.data_as,
                     self.data_as, -1, a_c,
                     0, 1, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec


    cdef vector_array_base rsub_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdSub(self.length,  a.data_as, self.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def __rsub__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.rsub_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.rsub_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef vector_array_base mul_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        cblas_daxpby(self.length, a_c, self.data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec


    cdef vector_array_base mul_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdMul(self.length, self.data_as, a.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec


    def __mul__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.mul_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.mul_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef void imul_scalar(vector_array_base self, double a):
        cblas_dscal(self.length, a, self.data_as, 1)
        return


    cdef void imul_vector(vector_array_base self, vector_array_base a):
        vdMul(self.length, self.data_as, a.data_as, self.data_as)

        return

    def __imul__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            vector_array_base.imul_vector(self, a_vec)
            return self
        if isinstance(a, float):
            a_scal = a
            vector_array_base.imul_scalar(self, a_scal)
            return self
        raise Exception("Unsupported type '"+str(type(a))+"'")


    def __rmul__(self, a):
        return self.__mul__(a)

    cdef vector_array_base pow_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        vdPowx(self.length, self.data_as, a,
               res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    cdef vector_array_base pow_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdPow(self.length, self.data_as,
              a.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec


    def __pow__(self, a,z):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.pow_vector(self, a_vec)

        if isinstance(a, float):
            a_scal = a
            return vector_array_base.pow_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef vector_array_base truediv_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_daxpby(self.length, 1/a, self.data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec


    cdef vector_array_base truediv_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdDiv(self.length, self.data_as, a.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def __truediv__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.truediv_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.truediv_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef vector_array_base rtruediv_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_daxpby(self.length, 1/a, self.data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec.data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    cdef vector_array_base rtruediv_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdDiv(self.length,  a.data_as, self.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def __rtruediv__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            return vector_array_base.rtruediv_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.rtruediv_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef void itruediv_scalar(vector_array_base self, double a):
        cdef double a_c = a
        cblas_daxpby(self.length, 1/a_c, self.data_as, 1, 0.0, self.data_as, 1)
        return


    cdef void itruediv_vector(vector_array_base self, vector_array_base a):
        vdDiv(self.length, self.data_as, a.data_as, self.data_as)
        return

    def __itruediv__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, vector_array_base):
            a_vec = a
            vector_array_base.itruediv_vector(self, a_vec)
            return self
        if isinstance(a, float):
            a_scal = a
            vector_array_base.itruediv_scalar(self, a_scal)
            return self
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef vector_array_base negate(vector_array_base self):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_daxpby(self.length, -1, self.data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def __neg__(self):
        return self.negate()

    cdef vector_array_base copy(vector_array_base self):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_dcopy(self.length, self.data_as, 1, res_ptr, 1)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def __pos__(self):
        return self.copy()

    cdef vector_array_base absolute(vector_array_base self):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdAbs(self.length, self.data_as, res_ptr)
        cdef vector_array_base res_vec = vector_array_base()
        res_vec.data = res
        res_vec.data_as = res_ptr
        res_vec.shape = self.shape
        res_vec.length = self.length
        return res_vec

    def to_numpy_array(self):
        """
        Return numpy array
        """
        return self.data



def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    vec = vector_array_c(param)
    if shape is not None:
        vec.shape = shape
    return vec




cdef vector_array_base vector_array_c(param):
    retval = vector_array_base()
    cdef np.ndarray[double, ndim=1] np_arr
    if isinstance(param, np.ndarray):
        np_arr =  param.copy().reshape((param.size,))
        retval.data = np_arr
        retval.data_as = &np_arr[0]
        retval.shape = param.shape
        retval.length = <int> np.prod(param.shape)
        return retval

    if isinstance(param, vector_array_base):
        np_arr = param.data.copy()
        retval.data = np_arr
        retval.data_as = &np_arr[0]
        retval.shape = param.shape
        retval.length = <int> np.prod(param.shape)
        return retval

    if isinstance(param, list):
        np_arr = np.array(param)
        retval.data = np_arr
        retval.data_as = &np_arr[0]
        retval.shape = (len(param),)
        retval.length = <int> len(param)

        return retval

    raise Exception("Type '"+str(type(param))+"' of param not supported")

def vector_array_zeros(shape):
    return vector_array_zeros_c(shape)

cdef vector_array_base vector_array_zeros_c(shape):
    """
    Return array of shape with zeros
    """
    cdef np.ndarray[double, ndim=1] np_arr
    retval = vector_array_base()
    retval.shape = shape
    retval.length = <int> np.prod(shape)
    np_arr = np.zeros(shape)
    retval.data = np_arr
    retval.data_as = &np_arr[0]
    return retval



def vector_array_zeros_like(data):
    """
    Return zero array of same shape as data
    """
    return vector_array_zeros(data.shape)

def vector_array_ones(shape):
    return vector_array_ones_c(shape)

cdef vector_array_base vector_array_ones_c(shape):
    """
    Return array of shape with ones
    """
    cdef np.ndarray[double, ndim=1] np_arr
    retval = vector_array_base()
    retval.shape = shape
    retval.length = <int> np.prod(shape)
    np_arr = np.ones(shape)
    retval.data = np_arr
    retval.data_as = &np_arr[0]
    return retval



