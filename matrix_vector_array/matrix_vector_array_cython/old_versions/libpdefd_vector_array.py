import numpy as np
import backend as b
import time
backend_str = b.get_backend()
if backend_str == 'scipy':
    try:
        from libpdefd.matrix_vector_array.libpdefd_vector_array_sp import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from libpdefd_vector_array_sp import *
elif backend_str == 'mkl':
    try:
        from libpdefd.matrix_vector_array.libpdefd_vector_array_mkl import *
    except:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from libpdefd_vector_array_mkl import *
else:
    raise Exception("Unsupported backend used")
