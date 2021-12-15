import cython
import numpy as np
cimport numpy as np
from ctypes import *
import sys
import os
sys.path.append(os.path.dirname(__file__))
from libc.stdlib cimport malloc, free
np.import_array() # initialize C API to call PyArray_SimpleNewFromData

mkl = cdll.LoadLibrary("libmkl_rt.so")


cdef extern from "mkl.h":

    cdef void cblas_dcopy(int length, double* x, int incx, double* y, int incy) nogil
    cdef void cblas_daxpy(int length, double a, double* x, int incx, double* y, int incy) nogil
    cdef void cblas_daxpby(int length, double a, double* x, int incx, double b, double* y, int incy) nogil

    cdef double* vdLinearFrac (int length, double* arr1, double* arr2, double fac1,
                               double scal1, double fac2, double scal2, double* res) nogil
    cdef void cblas_dsbmv(int layout, int uplo, int length, int k, double alpha, double* a, int lda, double* x,
                          int incx, double beta, double* y, int incy) nogil
    cdef void vdPow (int length, double* arr1, double* arr2, double* res) nogil
    cdef void vdPowx (int length, double* arr1, double scal, double* res) nogil
    cdef void vdDiv (int length, double* arr1, double* arr2, double* res) nogil
    cdef void vdAbs (const int n, const double* a, double* y) nogil
    cdef void cblas_dcopy(int length, double* arr1, int incx, double* res, int incy) nogil
    cdef void cblas_dscal(int length, double a, double* arr, int step) nogil
    cdef int cblas_idamax (const int n, const double *x, const int incx) nogil
    cdef int cblas_idamin (const int n, const double *x, const int incx) nogil
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
    def __cinit__(self):
        self.shape = ()
        self.length = 0

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._data_as[i]
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
            return self._data_as[j]

    def __setitem__(self, i, data):
        if isinstance(i, int):
            self._data_as[i] = data
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
            self._data_as[j] = data
        return
    cdef vector_array_base add_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        vdLinearFrac(self.length, self._data_as,
                     self._data_as, 1, a_c,
                     0, 1, res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec



    cdef vector_array_base add_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_dcopy(self.length, a._data_as, 1, res_ptr, 1)
        cblas_daxpy(self.length, 1, self._data_as, 1, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

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


    cdef void iadd_scalar(vector_array_base self, double a):
        cdef double a_c = a
        vdLinearFrac(self.length, self._data_as,
                     self._data_as, 1, a_c,
                     0, 1, self._data_as)
        return
    cdef void iadd_vector(vector_array_base self, vector_array_base a):
        cblas_dcopy(self.length, a._data_as, 1, self._data_as, 1)
        cblas_daxpy(self.length, 1, self._data_as, 1, self._data_as, 1)
        return

    def __iadd__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            vector_array_base.iadd_vector(self, a_vec)
            return self
        if isinstance(a, float):
            a_scal = a
            vector_array_base.iadd_scalar(self, a_scal)
            return self
        raise Exception("Unsupported type '"+str(type(a))+"'")

    def __radd__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.add_vector(a_vec, self)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.add_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef vector_array_base sub_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        vdLinearFrac(self.length, self._data_as,
                     self._data_as, 1, -a_c,
                     0, 1, res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    cdef vector_array_base sub_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_dcopy(self.length, self._data_as, 1, res_ptr, 1)
        cblas_daxpy(self.length, -1, a._data_as, 1, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

    def __sub__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.sub_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.sub_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef void isub_scalar(vector_array_base self, double a):
        cdef double a_c = a
        vdLinearFrac(self.length, self._data_as,
                     self._data_as, 1, -a_c,
                     0, 1, self._data_as)
        return


    cdef void isub_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_daxpy(self.length, -1, a._data_as, 1, self._data_as, 1)
        return

    def __isub__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
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
        vdLinearFrac(self.length, self._data_as,
                     self._data_as, -1, a_c,
                     0, 1, res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    cdef vector_array_base rsub_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_dcopy(self.length, a._data_as, 1, res_ptr, 1)
        cblas_daxpy(self.length, -1, self._data_as, 1, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

    def __rsub__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
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
        cblas_daxpby(self.length, a_c, self._data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    cdef vector_array_base mul_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_dsbmv(101, 122, self.length, 0, 1.0, self._data_as, 1, a._data_as, 1, 0.0,
                         res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    def __mul__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.mul_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.mul_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef void imul_scalar(vector_array_base self, double a):
        cdef double a_c = a
        cblas_dscal(self.length, a_c, self._data_as, 1)
        return


    cdef void imul_vector(vector_array_base self, vector_array_base a):
        cblas_dsbmv(101, 122, self.length, 0, 1.0, self._data_as, 1, a._data_as, 1, 0.0,
                    self._data_as, 1)
        return

    def __imul__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
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
        vdPowx(self.length, self._data_as, c_double(a),
               res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    cdef vector_array_base pow_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdPow(self.length, self._data_as,
                  a._data_as, res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    def __pow__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.pow_vector(self, a_vec)

        if isinstance(a, float):
            a_scal = a
            return vector_array_base.pow_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef vector_array_base truediv_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        cblas_daxpby(self.length, 1/a_c, self._data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    cdef vector_array_base truediv_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdDiv(self.length, self._data_as, a._data_as, res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

    def __truediv__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.truediv_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.truediv_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")


    cdef vector_array_base rtruediv_scalar(vector_array_base self, double a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cdef double a_c = a
        cblas_daxpby(self.length, 1/a_c, self._data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec


    cdef vector_array_base rtruediv_vector(vector_array_base self, vector_array_base a):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdDiv(self.length,  a._data_as, self._data_as,res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

    def __rtruediv__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
            a_vec = a
            return vector_array_base.rtruediv_vector(self, a_vec)
        if isinstance(a, float):
            a_scal = a
            return vector_array_base.rtruediv_scalar(self, a_scal)
        raise Exception("Unsupported type '"+str(type(a))+"'")

    cdef void itruediv_scalar(vector_array_base self, double a):
        cdef double a_c = a
        cblas_daxpby(self.length, 1/a_c, self._data_as, 1, 0.0, self._data_as, 1)
        return


    cdef void itruediv_vector(vector_array_base self, vector_array_base a):
        vdDiv(self.length, self._data_as, a._data_as, self._data_as)
        return

    def __itruediv__(self, a):
        cdef vector_array_base a_vec
        cdef double a_scal
        if isinstance(a, self.__class__):
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
        cblas_daxpby(self.length, -1, self._data_as, 1, 0.0, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

    def __neg__(self):
        return self.negate()

    cdef vector_array_base copy(vector_array_base self):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        cblas_dcopy(self.length, self._data_as, 1, res_ptr, 1)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

    def __pos__(self):
        return self.copy()

    cdef vector_array_base absolute(vector_array_base self):
        cdef np.ndarray[double, ndim=1] res = np.empty(self.length)
        cdef double* res_ptr = &res[0]
        vdAbs(self.length, self._data_as, res_ptr)
        cdef vector_array_base vec = vector_array_base()
        vec._data_as = res_ptr
        vec.length = self.length
        vec.shape = self.shape
        return vec

    def abs(self):
        return self.absolute()

    def reduce_min(self):
        ArrayType = c_double*self.length
        return min(np.frombuffer(ArrayType.from_address(addressof(self._data_as.contents))))


    def reduce_minabs(self):
        pos = mkl.cblas_idamin(self.length, self._data_as, 1)
        return abs((self._data_as)[pos])

    def reduce_max(self):
        ArrayType = c_double*self.length
        return max(np.frombuffer(ArrayType.from_address(addressof(self._data_as.contents))))

    def reduce_maxabs(self):
        pos = mkl.cblas_idamax(self.length, self._data_as, 1)
        return abs((self._data_as)[pos])

    def copy(self):
        res_np = np.empty(self.shape)
        res = res_np.ctypes.data_as(POINTER(c_double))
        mkl.cblas_dcopy(self.length, self._data_as, 1, res, 1)
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
        mkl.vdLinearFrac(self.c_length, self._data_as,
                         self._data_as, c_double(0), c_double(scalar_value),
                         c_double(0), c_double(1), self._data_as)
        return

    # def flatten(self):
    #     return self.__class__(self._data_as, shape=np.prod(self.shape))
    #
    # def num_elements(self):
    #     return self.length
    #
    # def __str__(self):
    #     retstr = "PYSMarray: "
    #     retstr += str(self.shape)
    #     return retstr
    #
    # def to_numpy_array(self):
    #     """
    #     Return numpy array
    #     """
    #     ArrayType = c_double*self.length
    #     return np.frombuffer(ArrayType.from_address(addressof(self._data_as.contents))).reshape(self.shape)


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    return vector_array_c(param)


cdef vector_array_base vector_array_c(param):
    cdef np.ndarray[double, ndim=1] param_flat
    cdef double* param_ptr
    cdef int length
    cdef vector_array_base res = vector_array_base()
    if isinstance(param, np.ndarray):
        param_flat = param.reshape((param.size,))
        param_ptr = &param_flat[0]
        length = <int> param.size
        res.length = length
        res.shape = param.shape
        res._data_as = param_ptr
        return res
    raise Exception("Type '"+str(type(param))+"' of param not supported")



# def vector_array_zeros(shape, dtype=None):
#     """
#     Return array of shape with zeros
#     """
#     retval = vector_array_base()
#     retval.shape = shape
#     retval.length = np.prod(shape)
#     retval.c_length = c_int(retval.length)
#     retval._data_as = np.zeros(shape).ctypes.data_as(POINTER(c_double))
#     return retval
#
#
#
# def vector_array_zeros_like(data, dtype=None):
#     """
#     Return zero array of same shape as data
#     """
#     return vector_array_zeros(data.shape, dtype=dtype)
#
#
# def vector_array_ones(shape, dtype=None):
#     """
#     Return array of shape with ones
#     """
#     retval = vector_array_base()
#     _data = np.ones(shape)
#     retval._data_as = _data.ctypes.data_as(POINTER(c_double))
#     retval.shape = shape
#     retval.length = np.prod(shape)
#     retval.c_length = c_int(retval.length)
#     return retval

