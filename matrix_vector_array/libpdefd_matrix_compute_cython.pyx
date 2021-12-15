
import numpy as np
import scipy.sparse as sparse
import sys
import os
import backend as b
"""
Use this hack to use these python files also without libpdefd
"""
backend_str = b.get_backend()

if backend_str == 'scipy':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_matrix_compute_sp import *
        #from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_np import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_python.libpdefd_matrix_compute_sp import *
        #from matrix_vector_array_python.libpdefd_vector_array_np import *
        sys.path.pop()
elif backend_str == 'mkl':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_matrix_compute_mkl import *
        #from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_mkl import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_python.libpdefd_matrix_compute_mkl import *
        #from matrix_vector_array_python.libpdefd_vector_array_mkl import *
        sys.path.pop()
elif backend_str in ['mkl_cython', 'mkl_cython_c', 'mkl_cython_malloc', 'mkl_blas_cython', 'mkl_blas_cython_c']:
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.libpdefd_matrix_compute_mkl import *
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.libpdefd_matrix_compute_mkl cimport *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.libpdefd_matrix_compute_mkl import *
        from matrix_vector_array_cython.libpdefd_matrix_compute_mkl cimport *
        sys.path.pop()
else:
    raise Exception("Unsupported backend")


