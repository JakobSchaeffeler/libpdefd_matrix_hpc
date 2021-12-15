# distutils: sources = vector.cpp
# distutils: include_dirs = .
# distutils: language = c++
cimport cvector
import numpy as np
cimport numpy as np
np.import_array() # initialize C API to call PyArray_SimpleNewFromData
from libc.stdlib cimport malloc, free
import time


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

    def __dealloc__(self):
        self.dealloc()

    cdef void dealloc(self):
        cvector.vector_free(self._vector)

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
        cdef cvector.Vector vec = cvector.add_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    cdef void iadd_vector(self, vector_array_base a):
        cvector.iadd_vector(self._vector, a._vector)
        return

    cdef vector_array_base add_scalar(self, double a):
        cdef cvector.Vector vec = cvector.add_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    cdef void iadd_scalar(self, double a):
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
        cdef cvector.Vector vec = cvector.mul_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    cdef vector_array_base mul_scalar(self, double a):
        cdef cvector.Vector vec = cvector.mul_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
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


    cdef void imul_vector(self, vector_array_base a):
        cvector.imul_vector(self._vector, a._vector)
        return

    cdef void imul_scalar(self, double a):
        cvector.imul_scalar(self._vector, a)
        return

    def __sub__(self, a):
        if isinstance(a, vector_array_base):
            return vector_array_base.sub_vector(self, a)
        elif isinstance(a, float):
            return vector_array_base.sub_scalar(self, a)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")


    cdef vector_array_base sub_vector(self, vector_array_base a):
        cdef cvector.Vector vec = cvector.sub_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    cdef vector_array_base sub_scalar(self, double a):
        cdef cvector.Vector vec = cvector.sub_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    def __isub__(self, a):
        if isinstance(a, vector_array_base):
            vector_array_base.isub_vector(self, a)
            return self
        elif isinstance(a, float):
            vector_array_base.isub_scalar(self, a)
            return self
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef void isub_vector(self, vector_array_base a):
        cvector.isub_vector(self._vector, a._vector)
        return

    cdef void isub_scalar(self, double a):
        cvector.isub_scalar(self._vector, a)
        return


    def __rsub__(self, a):
        if isinstance(a, vector_array_base):
            return vector_array_base.rsub_vector(self, a)
        elif isinstance(a, float):
            return vector_array_base.rsub_scalar(self, a)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")


    cdef vector_array_base rsub_vector(self, vector_array_base a):
        cdef cvector.Vector vec = cvector.rsub_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    cdef vector_array_base rsub_scalar(self, double a):
        cdef cvector.Vector vec = cvector.rsub_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    def __pow__(self, a, trash):
        if isinstance(a, vector_array_base):
            return vector_array_base.pow_vector(self, a)
        elif isinstance(a, float):
            return vector_array_base.pow_scalar(self, a)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")


    cdef vector_array_base pow_vector(self, vector_array_base a):
        cdef cvector.Vector vec = cvector.pow_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res


    cdef vector_array_base pow_scalar(self, double a):
        cdef cvector.Vector vec = cvector.pow_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    def __truediv__(self, a):
        if isinstance(a, vector_array_base):
            return vector_array_base.truediv_vector(self, a)
        elif isinstance(a, float):
            return vector_array_base.truediv_scalar(self, a)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef vector_array_base truediv_vector(self, vector_array_base a):
        cdef cvector.Vector vec = cvector.truediv_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res


    cdef vector_array_base truediv_scalar(self, double a):
        cdef cvector.Vector vec = cvector.truediv_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    def __rtruediv__(self, a):
        if isinstance(a, vector_array_base):
            return vector_array_base.rtruediv_vector(self, a)
        elif isinstance(a, float):
            return vector_array_base.rtruediv_scalar(self, a)
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef vector_array_base rtruediv_vector(self, vector_array_base a):
        cdef cvector.Vector vec = cvector.rtruediv_vector(self._vector, a._vector)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res


    cdef vector_array_base rtruediv_scalar(self, double a):
        cdef cvector.Vector vec = cvector.rtruediv_scalar(self._vector, a)
        cdef vector_array_base res = vector_array_base()
        res._vector = vec
        res.shape = self.shape
        return res

    def __itruediv__(self, a):
        if isinstance(a, vector_array_base):
            vector_array_base.itruediv_vector(self, a)
            return self
        elif isinstance(a, float):
            vector_array_base.itruediv_scalar(self, a)
            return self
        else:
            raise Exception("Unsupported type data '" + str(type(a)) + "'")

    cdef void itruediv_vector(self, vector_array_base a):
        cvector.itruediv_vector(self._vector, a._vector)
        return

    cdef void itruediv_scalar(self, double a):
        cvector.itruediv_scalar(self._vector, a)
        return

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return cvector.neg(self._vector)

    def get_elem(self):
        cdef double d = cvector.get_data(self._vector)[0]
        return d

    def copy(self):
        cdef vector_array_base res = vector_array_base()
        res._vector = cvector.copy(self._vector)
        return res

    def to_numpy_array(self):
        return vector_array_base.convert_to_numpy_array(self)

    cdef np.ndarray convert_to_numpy_array(self):
        cdef np.npy_intp dims = cvector.get_length(self._vector)
        cdef np.ndarray a = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_DOUBLE, <void*> cvector.get_data(self._vector))
        return a.reshape(self.shape)


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    return vector_array_c(param)


cdef vector_array_base vector_array_c(param):
    cdef np.ndarray[double, ndim=1] param_flat
    cdef double* param_ptr
    cdef int length
    cdef vector_array_base res = vector_array_base()
    cdef cvector.Vector vec
    if isinstance(param, np.ndarray):
        param_flat = param.reshape((param.size,))
        param_ptr = &param_flat[0]
        length = <int> param.size
        vec = cvector.vec_from_ptr(length, param_ptr)
        res.shape = param.shape
        res._vector = vec
        return res
    raise Exception("Type '"+str(type(param))+"' of param not supported")


def test_iadd_np_empty(N):
    cdef np.ndarray[double, ndim=1] x
    cdef double *x_ptr 
    cdef np.ndarray[double, ndim=1] y
    cdef double *y_ptr
    cdef np.ndarray[double, ndim=1] a
    cdef np.ndarray[double, ndim=1] b
    cdef double* param1 
    cdef double* param2
    for i in (24,):
        cvector.set_threads(i)
        t_alloc = 0 
        t_add = 0
        t_copy = 0
        for _ in range(0,100):
            x = np.random.rand(N)
            y = np.random.rand(N)
            x_ptr= &x[0]
            y_ptr= &y[0]
            start = time.time()
            a = np.empty(N)
            b = np.empty(N)
            end = time.time()
            t_alloc += end-start
            #print("allocation took", end-start, "seconds")
            param1 = &a[0]
            param2 = &b[0]
            start = time.time()
            cvector.copy_double(x_ptr, param1, N)
            cvector.copy_double(y_ptr, param2, N)
            end = time.time()
            t_copy += end-start
            start = time.time()
            for _ in range(0,10):
                cvector.iadd_double_vector(param1, param2, N)
            end = time.time()
            t_add += end-start
        print("t_alloc: ", t_alloc/100)
        print("t_copy: ", t_copy/100)
        print("t_add: ", t_add/1000)

def test_iadd_malloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1
    cdef double* param2

    for i in (24,):
        cvector.set_threads(i)
        x = np.random.rand(N)
        y = np.random.rand(N)
        x_ptr= &x[0]
        y_ptr= &y[0]
        param1 = cvector.malloc_vector(N)
        param2 = cvector.malloc_vector(N)
        cvector.copy_double(x_ptr, param1, N)
        cvector.copy_double(y_ptr, param2, N)
        print("{0:x}".format(<unsigned long>&param1[0]))
        print("{0:x}".format(<unsigned long>&param1[1]))
        for _ in range(1,10):
            cvector.iadd_double_vector(param1, param2, N)

        cvector.ptr_free(param1)
        cvector.ptr_free(param2)

def test_iadd_mkl_malloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1
    cdef double* param2
    for i in (24,):
        cvector.set_threads(i)
        for _ in range(5):
            x = np.random.rand(N)
            y = np.random.rand(N)
            x_ptr= &x[0]
            y_ptr= &y[0]
            param1 = cvector.mkl_malloc_vector(N)
            param2 = cvector.mkl_malloc_vector(N)
            cvector.copy_double(x_ptr, param1, N)
            cvector.copy_double(y_ptr, param2, N)
            for _ in range(1,10):
                cvector.iadd_double_vector(param1, param2, N)

            cvector.ptr_mkl_free(param1)
            cvector.ptr_mkl_free(param2)

def test_iadd_mkl_realloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1 = NULL
    cdef double* param2 = NULL
    for i in (24,):
        cvector.set_threads(i)
        x = np.random.rand(N)
        y = np.random.rand(N)
        x_ptr= &x[0]
        y_ptr= &y[0]
        param1 = cvector.mkl_realloc_vector(param1, N)
        param2 = cvector.mkl_realloc_vector(param2, N)
        cvector.copy_double(x_ptr, param1, N)
        cvector.copy_double(y_ptr, param2, N)
        for _ in range(1,10):
            cvector.iadd_double_vector(param1, param2, N)

def test_iadd_numa_alloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1 = NULL
    cdef double* param2 = NULL
    for i in (24,):
        cvector.set_threads(i)
        for _ in range(10):
            x = np.random.rand(N)
            y = np.random.rand(N)
            x_ptr= &x[0]
            y_ptr= &y[0]
            start = time.time()
            param1 = cvector.numa_alloc_vector(N)
            param2 = cvector.numa_alloc_vector(N)
            end = time.time()
            print("allocation took", end-start, "seconds")
            cvector.copy_double(x_ptr, param1, N)
            cvector.copy_double(y_ptr, param2, N)
            for _ in range(1,10):
                cvector.iadd_double_vector(param1, param2, N)

            cvector.numa_free_vector(param1, N)
            cvector.numa_free_vector(param2, N)

def test_iadd_alloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1 = NULL
    cdef double* param2 = NULL
    for i in (24,):
        cvector.set_threads(i)
        t_alloc = 0 
        t_add = 0
        t_copy = 0

        for _ in range(1,100):
            x = np.random.rand(N)
            y = np.random.rand(N)
            x_ptr= &x[0]
            y_ptr= &y[0]
            start = time.time()
            param1 = cvector.alloc_vector(N)
            param2 = cvector.alloc_vector(N)
            end = time.time()
            #print("allocation took", end-start, "seconds")
            t_alloc += end-start
            start = time.time()
            cvector.copy_double(x_ptr, param1, N)
            cvector.copy_double(y_ptr, param2, N)
            end = time.time()
            t_copy += end-start
            start = time.time()
            for _ in range(1,10):
                cvector.iadd_double_vector(param1, param2, N)
            end = time.time()
            t_add += end-start
 
            cvector.alloc_free(param2, N)
            cvector.alloc_free(param1, N)
        print("t_alloc: ", t_alloc/100)
        print("t_copy: ", t_copy/100)
        print("t_add: ", t_add/1000)

           


def test_imul_np_empty(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]

    cdef np.ndarray[double, ndim=1] a = np.empty(N)
    cdef double* param1 = &a[0]
    for i in (1,24,):
        cvector.set_threads(i)
        x = np.random.rand(N)
        x_ptr= &x[0]
        a = np.empty(N)
        param1 = &a[0]
        cvector.copy_double(x_ptr, param1, N)
        for _ in range(1,10):
            cvector.imul_double_scalar(param1, np.random.rand(), N)


def test_imul_malloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1
    cdef double* param2

    for i in (1,24,):
        cvector.set_threads(i)
        x = np.random.rand(N)
        y = np.random.rand(N)
        x_ptr= &x[0]
        y_ptr= &y[0]
        param1 = cvector.malloc_vector(N)
        param2 = cvector.malloc_vector(N)
        cvector.copy_double(x_ptr, param1, N)
        cvector.copy_double(y_ptr, param2, N)
        print("{0:x}".format(<unsigned long>&param1[0]))
        print("{0:x}".format(<unsigned long>&param1[1]))
        for _ in range(1,10):
            cvector.imul_double_scalar(param1, np.random.rand(), N)

        cvector.ptr_free(param1)
        cvector.ptr_free(param2)

def test_imul_mkl_malloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1
    cdef double* param2
    for i in (1,24,):
        cvector.set_threads(i)
        x = np.random.rand(N)
        y = np.random.rand(N)
        x_ptr= &x[0]
        y_ptr= &y[0]
        param1 = cvector.mkl_malloc_vector(N)
        param2 = cvector.mkl_malloc_vector(N)
        cvector.copy_double(x_ptr, param1, N)
        cvector.copy_double(y_ptr, param2, N)
        for _ in range(1,10):
            cvector.imul_double_scalar(param1, np.random.rand(), N)

        cvector.ptr_mkl_free(param1)
        cvector.ptr_mkl_free(param2)

def test_imul_mkl_realloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1 = NULL
    cdef double* param2 = NULL
    for i in (1,24,):
        cvector.set_threads(i)
        x = np.random.rand(N)
        y = np.random.rand(N)
        x_ptr= &x[0]
        y_ptr= &y[0]
        param1 = cvector.mkl_realloc_vector(param1, N)
        param2 = cvector.mkl_realloc_vector(param2, N)
        cvector.copy_double(x_ptr, param1, N)
        cvector.copy_double(y_ptr, param2, N)
        for _ in range(1,10):
            cvector.imul_double_scalar(param1, np.random.rand(), N)

def test_imul_numa_alloc(N):
    cdef np.ndarray[double, ndim=1] x = np.random.rand(N)
    cdef double *x_ptr= &x[0]
    cdef np.ndarray[double, ndim=1] y = np.random.rand(N)
    cdef double *y_ptr= &y[0]
    cdef double* param1 = NULL
    cdef double* param2 = NULL
    for i in (1,24,):
        cvector.set_threads(i)
        x = np.random.rand(N)
        y = np.random.rand(N)
        x_ptr= &x[0]
        y_ptr= &y[0]
        param1 = cvector.numa_alloc_vector(N)
        param2 = cvector.numa_alloc_vector(N)
        cvector.copy_double(x_ptr, param1, N)
        cvector.copy_double(y_ptr, param2, N)
        for _ in range(1,10):
            cvector.imul_double_scalar(param1, np.random.rand(), N)

        cvector.numa_free_vector(param1, N)
        cvector.numa_free_vector(param2, N)
