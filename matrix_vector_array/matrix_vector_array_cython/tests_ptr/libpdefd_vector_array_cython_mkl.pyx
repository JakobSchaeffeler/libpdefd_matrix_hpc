# distutils: sources = vector.c
# distutils: include_dirs = .

cimport cvector
import numpy as np
cimport numpy as np
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
    cdef cvector.Vector* _vector
    cdef int length

    def __init__(self, cvector.Vector* vec):
        self._vector = vec

    def __dealloc__(self):
        cvector.vector_free(self._vector)

    cdef add_vec(self, vector_array_base a):
            cdef cvector.Vector *vec = cvector.add_vector(self._vector, a._vector)
            return self.__class__(vec)

    cdef add_scal(self, double a):
            cdef cvector.Vector *vec = cvector.add_scalar(self._vector, a)
            return self.__class__(vec)


def vector_array(param, dtype=None, shape = None, *args, **kwargs):
    cdef np.ndarray[double, ndim=1] param_flat
    cdef double* param_ptr
    cdef int length
    cdef cvector.Vector vec = cvector.vector_alloc()
    if isinstance(param, np.ndarray):
        param_flat = param.reshape((len(param),1))
        param_ptr = &param_flat[0]
        vec.data = param_ptr
        length = <int> len(param)
        vec.length = length
        return vec

    if isinstance(param, vector_array_base):
        return vector_array_base(param._vector)

    raise Exception("Type '"+str(type(param))+"' of param not supported")

