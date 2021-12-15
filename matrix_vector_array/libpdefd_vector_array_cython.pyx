import numpy as np
import backend as b
import sys, os
import time
backend_ = ['scipy', 'scipy_cython', 'mkl', 'mkl_cython', 'mkl_cython_c', 'mkl_blas', 'mkl_cython_malloc', 'mkl_blas_cython', 'mkl_blas_cython_c']


backend_str = b.get_backend()

sys.path.append(os.path.dirname(__file__))


if backend_str == 'scipy':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_np import *
    except:
        from matrix_vector_array_python.libpdefd_vector_array_np import *
elif backend_str == 'mkl':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_mkl import *
    except:
        from matrix_vector_array_python.libpdefd_vector_array_mkl import *
elif backend_str == 'mkl_blas':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_blas import *
    except:
        from matrix_vector_array_python.libpdefd_vector_array_blas import *

elif backend_str == 'mkl_cython':
    from matrix_vector_array_cython.mkl.libpdefd_vector_array_cython_mkl cimport *
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl.libpdefd_vector_array_cython_mkl import *
    except:
        from matrix_vector_array_cython.mkl.libpdefd_vector_array_cython_mkl import *

elif backend_str == 'scipy_cython':
    from matrix_vector_array_cython.np.libpdefd_vector_array_cython_np cimport *
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.np.libpdefd_vector_array_cython_np import *
    except:
        from matrix_vector_array_cython.np.libpdefd_vector_array_cython_np import *

elif backend_str == 'mkl_cython_c':
    from matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c cimport *
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *
    except:
        from matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *

elif backend_str == 'mkl_cython_malloc':
    from matrix_vector_array_cython.mkl_malloc.libpdefd_vector_array_cython_mkl_malloc cimport *
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_malloc.libpdefd_vector_array_cython_mkl_malloc import *
    except:
        from matrix_vector_array_cython.mkl_malloc.libpdefd_vector_array_cython_mkl_malloc import *

elif backend_str == 'mkl_cython_c':
    from matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c cimport *
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *
    except:
        from matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *

elif backend_str == 'mkl_blas_cython':
    from matrix_vector_array_cython.mkl_blas.libpdefd_vector_array_cython_blas cimport *
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas.libpdefd_vector_array_cython_blas import *
    except:
        from matrix_vector_array_cython.mkl_blas.libpdefd_vector_array_cython_blas import *
elif backend_str == 'mkl_blas_cython_c':
    from matrix_vector_array_cython.mkl_blas_c.libpdefd_vector_array_cython_mkl_blas_c cimport *
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas_c.libpdefd_vector_array_cython_mkl_blas_c import *
    except:
        from matrix_vector_array_cython.mkl_blas_c.libpdefd_vector_array_cython_mkl_blas_c import *
else:
    raise Exception("Unsupported backend used")
