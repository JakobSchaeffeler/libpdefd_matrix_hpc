
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
        from libpdefd.matrix_vector_array.libpdefd_matrix_compute_sp import *
        from libpdefd.matrix_vector_array.libpdefd_vector_array_sp import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from libpdefd_matrix_compute_sp import *
        from libpdefd_vector_array_sp import *
        sys.path.pop()
elif backend_str == 'mkl':
    try:
        from libpdefd.matrix_vector_array.libpdefd_matrix_compute_mkl import *
        from libpdefd.matrix_vector_array.libpdefd_vector_array_mkl import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from libpdefd_matrix_compute_mkl import *
        from libpdefd_vector_array_mkl import *
        sys.path.pop()
else:
    raise Exception("Unsupported backend")


