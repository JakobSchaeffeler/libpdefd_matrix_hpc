import numpy as np
import backend as b
import time
backend_ = ['scipy', 'scipy_cython', 'mkl', 'mkl_cython', 'mkl_cython_c', 'mkl_blas_cython_cpp', 'mkl_blas', 'mkl_cython_naiv', 'mkl_cython_malloc', 'mkl_blas_cython', 'mkl_blas_cython_c']


version_cython_ = 'mkl_blas_cython_typed', 'mkl_blas_cython_typed_cdef'




backend_str = b.get_backend()
if backend_str == 'scipy':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_np import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_python.libpdefd_vector_array_np import *
elif backend_str == 'mkl':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_mkl import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_python.libpdefd_vector_array_mkl import *
elif backend_str == 'mkl_blas':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_python.libpdefd_vector_array_mkl_blas import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_python.libpdefd_vector_array_mkl_blas import *
elif backend_str == 'scipy_cython':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.np.libpdefd_vector_array_cython_np import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.np.libpdefd_vector_array_cython_np import *
elif backend_str == 'mkl_cython_naiv':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_naiv.libpdefd_vector_array_cython_mkl import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_naiv.libpdefd_vector_array_cython_mkl import *

elif backend_str == 'mkl_cython_typed':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_typed.libpdefd_vector_array_cython_mkl_typed import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_typed.libpdefd_vector_array_cython_mkl_typed import *

elif backend_str == 'mkl_cython_typed_cdef':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_typed_cdef.libpdefd_vector_array_cython_mkl_typed_cdef import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_typed_cdef.libpdefd_vector_array_cython_mkl_typed_cdef import *

elif backend_str == 'mkl_blas_cython_naiv':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas_naiv.libpdefd_vector_array_cython_mkl_blas import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_blas_naiv.libpdefd_vector_array_cython_mkl_blas import *

elif backend_str == 'mkl_blas_cython_typed':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas_typed.libpdefd_vector_array_cython_mkl_blas import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_blas_typed.libpdefd_vector_array_cython_mkl_blas import *

elif backend_str == 'mkl_blas_cython_typed_cdef':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas_typed_cdef.libpdefd_vector_array_cython_mkl_blas import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_blas_typed_cdef.libpdefd_vector_array_cython_mkl_blas import *




elif backend_str == 'mkl_cython_c':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *
elif backend_str == 'mkl_cython_malloc':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_malloc.libpdefd_vector_array_cython_mkl_malloc import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_malloc.libpdefd_vector_array_cython_mkl_malloc import *
elif backend_str == 'mkl_cython_c':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_c.libpdefd_vector_array_cython_mkl_c import *

elif backend_str == 'mkl_cython_typed':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_typed.libpdefd_vector_array_cython_mkl_typed import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_typed.libpdefd_vector_array_cython_mkl_typed import *

elif backend_str == 'mkl_cython_blas':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas_typed.libpdefd_vector_array_cython_blas import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_blas_typed.libpdefd_vector_array_cython_mkl_blas import *
elif backend_str == 'mkl_blas_cython_cpp':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas_cpp_alloc.libpdefd_vector_array_cython_mkl_blas_cpp import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_blas_cpp_alloc.libpdefd_vector_array_cython_mkl_blas_cpp import *
elif backend_str == 'mkl_cython_cpp':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_cpp_alloc.libpdefd_vector_array_cython_mkl_cpp_alloc import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_cpp_alloc.libpdefd_vector_array_cython_mkl_cpp_alloc import *



elif backend_str == 'mkl_blas_cython_c':
    try:
        from libpdefd.matrix_vector_array.matrix_vector_array_cython.mkl_blas_c.libpdefd_vector_array_cython_mkl_blas_c import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from matrix_vector_array_cython.mkl_blas_c.libpdefd_vector_array_cython_mkl_blas_c import *
else:
    print(backend_str)
    raise Exception("Unsupported backend used")
