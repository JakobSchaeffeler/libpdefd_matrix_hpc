
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import sys, os
from ctypes import *

sys.path.append(os.path.dirname(__file__))
import libpdefd_matrix_compute

mkl = cdll.LoadLibrary("libmkl_rt.so")


"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.matrix_vector_array.libpdefd_matrix_setup as libpdefd_matrix_setup
    import libpdefd.matrix_vector_array_mkl.libpdefd_vector_array as libpdefd_vector_array
except:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import libpdefd_vector_array_mkl as libpdefd_vector_array
    import libpdefd_matrix_setup as libpdefd_matrix_setup
    sys.path.pop()




"""
This module implements the compute class for sparse matrices

Hence, this should be highly optimized
"""


class _sparse_matrix(Structure):
    pass


class matrix_descr(Structure):
    _fields_ = [("sparse_matrix_type_t", c_int),
                ("sparse_fill_mode_t", c_int),
                ("sparse_diag_type_t", c_int)]

    def __init__(self, sparse_matrix_type_t=20, sparse_fill_mode_t=0, sparse_diag_type_t=0):
        super(matrix_descr, self).__init__(sparse_matrix_type_t, sparse_fill_mode_t, sparse_diag_type_t)


matr_dec = matrix_descr()


cdef class matrix_sparse:
    """
    This class is using a format which is suited for fast matrix-vector multiplications.
    """
    cdef double* _matrix_mkl
    cdef int length

    def __init__(self,data = None, shape = None):
        self._matrix_mkl = POINTER(_sparse_matrix)()
        self.setup(data, shape)

    def setup(self, data, shape = None):
        if isinstance(data, np.ndarray):
            matrix_csr = sparse.csr_matrix(data)
            self.shape = matrix_csr.shape
        elif isinstance(data, sparse.csr_matrix):
            matrix_csr = data
            self.shape = matrix_csr.shape
        elif isinstance(data, sparse.lil_matrix):
            matrix_csr = sparse.csr_matrix(data)
            self.shape = matrix_csr.shape
        elif isinstance(data, libpdefd_matrix_setup.matrix_sparse):
            matrix_csr = sparse.csr_matrix(data._matrix_lil)
            self.shape = matrix_csr.shape
        #elif isinstance(data, libpdefd_matrix_compute_np.matrix_sparse):
        #    self.shape = self._matrix_csr.shape
        #    matrix_csr = data._matrix_csr
        elif isinstance(data, POINTER(_sparse_matrix)):
            mkl.mkl_sparse_destroy(self._matrix_mkl)
            mkl.mkl_sparse_copy(data, matrix_descr(), byref(self._matrix_mkl))
            self.shape = shape
            return
        elif isinstance(data, matrix_sparse):
            self.shape = data.shape
            mkl.mkl_sparse_destroy(self._matrix_mkl)
            mkl.mkl_sparse_copy(data._matrix_mkl, matrix_descr(), byref(self._matrix_mkl))
            return
        else:
            raise Exception("Unsupported type data '" + str(type(data)) + "'")
        matrix_mkl = POINTER(_sparse_matrix)()
        rows_start = matrix_csr.indptr[0:-1].ctypes.data_as(POINTER(c_int))
        rows_end = matrix_csr.indptr[1:].ctypes.data_as(POINTER(c_int))
        col_indx = matrix_csr.indices.ctypes.data_as(POINTER(c_int))
        values = matrix_csr.data.ctypes.data_as(POINTER(c_double))
        suc = mkl.mkl_sparse_d_create_csr(byref(matrix_mkl), 0, matrix_csr.shape[0], matrix_csr.shape[1],
                                          rows_start,
                                          rows_end,
                                          col_indx,
                                          values)
        if suc != 0:
            raise Exception("Creation of MKL matrix failed")
        self._matrix_mkl = matrix_mkl
        self.rows_start = rows_start
        self.rows_end = rows_end
        self.col_indx = col_indx
        self.values = values
        return


    def __iadd__(self, data):
        if isinstance(data, matrix_sparse):
            matrix = POINTER(_sparse_matrix)()
            suc = mkl.mkl_sparse_d_add(10, self._matrix_mkl, c_double(1), data._matrix_mkl, byref(matrix))
            if suc != 0:
                print(suc)
                raise Exception("iadd failed")
            mkl.mkl_sparse_destroy(self._matrix_mkl)
            self._matrix_mkl = matrix
            return

        raise Exception("Unsupported type data '" + str(type(data)) + "'")


    def __str__(self):
        retstr = ""
        retstr += "PYSMmatrixcompute: "
        retstr += " shape: "+str(self.shape)
        retstr += "\n"
        retstr += str(self.to_numpy_array())
        return retstr


    def dot__DEPRECATED(self, data):
        """
        Compute
            M*x
        by reshaping 'x' so that it fits the number of rows in 'M'
        """


        if isinstance(data, libpdefd_vector_array.vector_array_base):
            """
            If input is libpdefd_array return libpdefd_array
            """
            m = self.shape[0]
            y = np.zeros(m)

            suc = mkl.mkl_sparse_d_mv(10, c_double(1), self._matrix_mkl,
                                      matr_dec, data._data_as,
                                      c_double(0), y.ctypes.data_as(POINTER(c_double)))

            if suc != 0:
                raise Exception("Computing dot product of sparse mkl matrix and libpdefd_array failed")

            """
            only convert pointer if necessary via: np.fromiter(y_pointer, dtype=np.double, count=m)
            """
            return libpdefd_vector_array.vector_array(y)

        if isinstance(data, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
            m = self.shape[0]
            y = np.zeros(m)

            suc = mkl.mkl_sparse_d_mv(10, c_double(1), self._matrix_mkl,
                                      matr_dec, data._data_as,
                                      c_double(0), y.ctypes.data_as(POINTER(c_double)))

            if suc != 0:
                raise Exception("Computing dot product of sparse mkl matrix and libpdefd_array failed")

            return y

        if isinstance(data, libpdefd_matrix_compute.matrix_sparse):
            """
            Do not use matrix_sparse_setup for C, it's very slow
            """
            matrix_mkl = POINTER(_sparse_matrix)()
            ret = mkl.mkl_sparse_spmm(10, self._matrix_mkl, data._matrix_mkl, byref(matrix_mkl))

            if ret != 0:
                print(ret)
                raise Exception("Computing dot product of two MKL sparse matrices failed")

            C = libpdefd_matrix_compute.matrix_sparse(matrix_mkl)
            mkl.mkl_sparse_destroy(matrix_mkl)
            mkl.mkl_free_buffers()
            return C

        raise Exception("Unsupported type data '" + str(type(data)) + "'")


    def dot_add__DEPRECATED(self, x, c):
        """
        Compute
            M*x + c
        """


        if type(x) != type(c):
            raise Exception("x and c must have same type")

        if isinstance(x, libpdefd_vector_array.vector_array_base):
            cs = c.copy()

            suc = mkl.mkl_sparse_d_mv(10, c_double(1), self._matrix_mkl,
                                      matrix_descr(), x._data_as,
                                      c_double(1), cs._data_as)
            if suc != 0:
                raise Exception("Computing dot product of sparse mkl matrix and libpdefd_array failed")

            return cs

        if isinstance(x, np.ndarray) and False:
            cs = c.copy()
            suc = mkl.mkl_sparse_d_mv(10, c_double(1), self._matrix_mkl,
                                      matrix_descr(), x.ctypes.data_as(POINTER(c_double)),
                                      c_double(1), cs.ctypes.data_as(POINTER(c_double)))
            if suc != 0:
                raise Exception("Computing dot product of sparse mkl matrix and libpdefd_array failed")

            return cs

        if isinstance(x, matrix_sparse):
            matrix_mkl = POINTER(_sparse_matrix)()
            ret = mkl.mkl_sparse_spmm(10, self._matrix_mkl, x._matrix_mkl, byref(matrix_mkl))

            if ret != 0:
                print(ret)
                raise Exception("Computing dot product of two MKL sparse matrices failed")
            C = matrix_sparse(matrix_mkl)
            mkl.mkl_sparse_destroy(matrix_mkl)
            C.__iadd__(c)
            return C

        raise Exception("Unsupported type data '" + str(type(x)) + "'")


    def dot_add_reshape(self, x, c, dst_shape):
        """
        Compute
            M*x + c
        by reshaping 'x' so that it fits the number of rows in 'M'.

        Reshape the output to fit dst_shape.
        """
        if type(x) != type(c):
            raise Exception("x and c must have same type")

        if isinstance(x, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
        if isinstance(x, libpdefd_vector_array.vector_array_base):
            cs = c.copy()
            suc = mkl.mkl_sparse_d_mv(10, c_double(1), self._matrix_mkl, matrix_descr(),x._data_as,
                                      c_double(1), cs._data_as)
            if suc != 0:
                raise Exception("Computing dot product of sparse mkl matrix and libpdefd_array failed")
            cs.shape = dst_shape
            return cs

        if isinstance(x, matrix_sparse):
            raise Exception("This case shouldn't exist")

        raise Exception("Unsupported type data '" + str(type(x)) + "' for x")

    def to_numpy_array(self):
        rows = c_int()
        cols = c_int()
        indexing = c_int()
        row_start = POINTER(c_int)()
        row_end = POINTER(c_int)()
        col_indx = POINTER(c_int)()
        values = POINTER(c_double)()

        suc = mkl.mkl_sparse_d_export_csr(self._matrix_mkl, byref(indexing), byref(rows), byref(cols), byref(row_start),
                                          byref(row_end), byref(col_indx), byref(values))
        if suc != 0:
            print(suc)
            raise Exception("Conversion from mkl_sparse to scipy sparse csr matrix failed")

        a_csr = sparse.csr_matrix((rows.value, cols.value))
        # TODO try with conversion as in vector_array
        row_start_np = np.fromiter(row_start, dtype=int, count=rows.value)
        row_end_np = np.fromiter(row_end, dtype=int, count=rows.value)
        col_indx_np = np.fromiter(col_indx, dtype=int, count=row_end_np[rows.value - 1])
        values_np = np.fromiter(values, dtype=c_double, count=row_end_np[rows.value - 1])

        row_start_np = np.append(row_start_np, row_end_np[-1])

        a_csr.indptr = row_start_np
        a_csr.indices = col_indx_np
        a_csr.data = values_np
        return a_csr.toarray()

    def set_mkl(self, matrix):
        matrix_mkl = POINTER(_sparse_matrix)()
        rows_start = matrix.indptr[0:-1].ctypes.data_as(POINTER(c_int))
        rows_end = matrix.indptr[1:].ctypes.data_as(POINTER(c_int))
        col_indx = matrix.indices.ctypes.data_as(POINTER(c_int))
        values = matrix.data.ctypes.data_as(POINTER(c_double))
        ref_matr = byref(matrix_mkl)
        suc = mkl.mkl_sparse_d_create_csr(ref_matr, 0, matrix.shape[0], matrix.shape[1],
                                          rows_start,
                                          rows_end,
                                          col_indx,
                                          values)
        if suc != 0:
            raise Exception("Creation of MKL matrix failed")
        self._matrix_mkl = matrix_mkl
        mkl.mkl_free_buffers()
        return

