# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse as sparse
import sys, os

a, trash = os.path.split(os.path.abspath(__file__))

path = os.path.join(a, 'mkl_typed')
sys.path.append(path)

from libpdefd_vector_array_cython_mkl_typed cimport vector_array_base as vector_array_base_typed
sys.path.pop()
"""
path = os.path.join(a, 'mkl_naiv')
sys.path.append(path)
from libpdefd_vector_array_cython_mkl_naiv cimport vector_array_base as vector_array_base_naiv
sys.path.pop()
"""

path = os.path.join(a, 'mkl_malloc')
sys.path.append(path)
from libpdefd_vector_array_cython_mkl_malloc cimport vector_array_base as vector_array_base_malloc
sys.path.pop()
path = os.path.join(a, 'mkl_cpp_alloc')
sys.path.append(path)
from libpdefd_vector_array_cython_mkl_cpp_alloc cimport vector_array_base as vector_array_base_cpp_alloc
sys.path.pop()

a, trash = os.path.split(a)
sys.path.append(a)
import libpdefd_matrix_setup




cdef class matrix_sparse:
    """
    This class is using a format which is suited for fast matrix-vector multiplications.
    """


    def __init__(self,data = None, shape = None):
        if data is not None:
            self.setup(data, shape)
        else:
            self.setup_empty(shape)

    def __dealloc__(self):
        mkl_sparse_destroy(self._matrix_mkl)


    cdef setup_empty(self, tuple shape):
        cdef matrix_sparse sparse_matr
        cdef sparse_matrix_t matr_t
        cdef matrix_descr descr = matrix_descr()
        self.shape = shape


    cdef setup(self, data, tuple shape):
        cdef matrix_sparse sparse_matr
        cdef sparse_matrix_t matr_t
        cdef matrix_descr descr = matrix_descr()

        if isinstance(data, np.ndarray):
            matrix_csr = sparse.csr_matrix(data)
            shape = matrix_csr.shape
        elif isinstance(data, sparse.csr_matrix):
            matrix_csr = data
            shape = matrix_csr.shape
        elif isinstance(data, sparse.lil_matrix):
            matrix_csr = sparse.csr_matrix(data)
            shape = matrix_csr.shape
        elif isinstance(data, libpdefd_matrix_setup.matrix_sparse):
            matrix_csr = sparse.csr_matrix(data._matrix_lil)
            shape = matrix_csr.shape
        elif isinstance(data, matrix_sparse):
            sparse_matr = data
            self.shape = sparse_matr.shape
            mkl_sparse_destroy(self._matrix_mkl)
            descr.type = SPARSE_MATRIX_TYPE_GENERAL
            descr.mode = SPARSE_FILL_MODE_LOWER
            descr.diag = SPARSE_DIAG_NON_UNIT
            mkl_sparse_copy(sparse_matr._matrix_mkl, descr, &self._matrix_mkl)
            return

        else:
            raise Exception("Unsupported type data '" + str(type(data)) + "'")

        """
        create memory view of csr components needed for mkl csr creation
        """
        cdef MKL_INT[:] indptr = matrix_csr.indptr
        cdef MKL_INT[:] indices = matrix_csr.indices
        cdef double[:] values = matrix_csr.data

        cdef sparse_matrix_t matrix_mkl

        suc = mkl_sparse_d_create_csr(&matrix_mkl, SPARSE_INDEX_BASE_ZERO, <MKL_INT> matrix_csr.shape[0], <MKL_INT> matrix_csr.shape[1],
                                    &indptr[0],
                                    &indptr[1],
                                    &indices[0],
                                    &values[0])
        if suc != 0:
            print(suc)
            raise Exception("Creation of MKL matrix failed")
        self._matrix_mkl = matrix_mkl
        self.values = values
        self.indptr = indptr
        self.indices = indices
        return

    def __iadd__(self, data):
        if(isinstance(data, matrix_sparse)):
            self.iadd(data)
        else:
            raise Exception("Matrix iadd with unsupported data typed")
        return self

    cdef void iadd(matrix_sparse self, matrix_sparse data):
        self._matrix_mkl = cmatrix.add(self._matrix_mkl, data._matrix_mkl)

    def __str__(self):
        cdef np.npy_intp dims = self.shape_len
        cdef np.ndarray a = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT, <void*> self.shape)
        retstr = ""
        retstr += "PYSMmatrixcompute: "
        retstr += " shape: "+str(a)
        retstr += "\n"
        retstr += str(self.to_numpy_array())
        return retstr

    def shape(self):
        return self.shape

    def dot_add_reshape(self, x, c, dst_shape):
        """
        Compute
            M*x + c
        by reshaping 'x' so that it fits the number of rows in 'M'.

        Reshape the output to fit dst_shape.
        """
        print(type(x))

        if type(x) != type(c):
            raise Exception("x and c must have same type")

        print(type(vector_array_base_malloc(np.random.rand())))
        #if isinstance(x, libpdefd_vector_array_cython_mkl.vector_array_base):
        #    return self.dot_add_reshape_array_base_mkl(x,c,dst_shape)
        if isinstance(x, vector_array_base_typed):
            return self.dot_add_reshape_array_base_mkl_typed(x,c,dst_shape)
        elif isinstance(x, vector_array_base_malloc):
            return self.dot_add_reshape_array_base_mkl_malloc(x,c,dst_shape)
        elif isinstance(x, vector_array_base_cpp_alloc):
            return self.dot_add_reshape_array_base_mkl_cpp_alloc(x,c,dst_shape)

        if isinstance(x, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """





        if isinstance(x, matrix_sparse):
            raise Exception("This case shouldn't exist")

        raise Exception("Unsupported type data '" + str(type(x)) + "' for x")


    """
    cdef vector_array_base_naiv dot_add_reshape_array_base_mkl(self, vector_array_base_naiv x, vector_array_base_naiv c, tuple dst_shape):
        raise Exception("Matrix operations for base version in cython not supported")

        cdef matrix_descr descr = matrix_descr()
        descr.type = SPARSE_MATRIX_TYPE_GENERAL
        descr.mode = SPARSE_FILL_MODE_LOWER
        descr.diag = SPARSE_DIAG_NON_UNIT
        cdef vector_array_base_naiv cs = c.copy()
        suc = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, <double> 1.0, self._matrix_mkl, descr, x.data_as,
                              <double> 1.0, cs.data_as)
        if suc != 0:
            print(suc)
            raise Exception("Computing dot product of sparse mkl matrix and libpdefd_array failed")
        cs.shape = dst_shape
        return cs
    """


    cdef vector_array_base_typed dot_add_reshape_array_base_mkl_typed(self, vector_array_base_typed x, vector_array_base_typed c, tuple dst_shape):
        cdef matrix_descr descr = matrix_descr()
        descr.type = SPARSE_MATRIX_TYPE_GENERAL
        descr.mode = SPARSE_FILL_MODE_LOWER
        descr.diag = SPARSE_DIAG_NON_UNIT
        cdef vector_array_base_typed cs = c.copy()
        suc = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, <double> 1.0, self._matrix_mkl, descr, x.data_as,
                              <double> 1.0, cs.data_as)
        if suc != 0:
            print(suc)
            raise Exception("Computing dot product of sparse mkl matrix and libpdefd_array failed")
        cs.shape = dst_shape
        return cs

    cdef vector_array_base_malloc dot_add_reshape_array_base_mkl_malloc(self, vector_array_base_malloc x, vector_array_base_malloc c, tuple dst_shape):
        cdef vector_array_base_malloc cs = c.copy()
        cmatrix.dot_add(self._matrix_mkl, x.data_as, cs.data_as)
        cs.shape = dst_shape
        return cs

    cdef vector_array_base_cpp_alloc dot_add_reshape_array_base_mkl_cpp_alloc(self, vector_array_base_cpp_alloc x, vector_array_base_cpp_alloc c, tuple dst_shape):
        cdef vector_array_base_cpp_alloc cs = c.copy()
        cmatrix.dot_add(self._matrix_mkl, x.get_data(), cs.get_data() )
        cs.shape = dst_shape
        return cs


    def to_numpy_array(self):
        cdef MKL_INT rows
        cdef MKL_INT cols
        cdef sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO
        cdef MKL_INT* row_start
        cdef MKL_INT* row_end
        cdef MKL_INT* col_indx
        cdef double* values

        suc = mkl_sparse_d_export_csr(self._matrix_mkl, &indexing, &rows, &cols, &row_start,
                                          &row_end, &col_indx, &values)
        if suc != 0:
            print(suc)
            raise Exception("Conversion from mkl_sparse to scipy sparse csr matrix failed")


        a_csr = sparse.csr_matrix((rows, cols))
        cdef int nnz = row_start[rows]
        data = np.asarray(<double[:nnz]> values)
        indices = np.asarray(<int[:nnz]> col_indx)
        indptr = np.empty(rows, dtype=np.int32)
        indptr = np.asarray(<int[:rows]> row_start)
        indptr = np.append(indptr,nnz)

        a_csr.data = data
        a_csr.indices = indices
        a_csr.indptr = indptr

        return a_csr.toarray()




