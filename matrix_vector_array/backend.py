import os
global backend
backend = None

backend_ = ['scipy', 'scipy_cython', 'mkl', 'mkl_cython', 'mkl_cython_c', 'mkl_cython_naiv', 'mkl_blas_cython_c', 'mkl_blas', 'mkl_cython_malloc', 'mkl_cython_typed','mkl_cython_blas', 'mkl_blas_cython_c', 'mkl_blas_cython_cpp', 'mkl_cython_cpp']

def __init__():
    global backend
    if 'PYSM_BACKEND' in os.environ:
        new_backend = os.environ['PYSM_BACKEND']
        if new_backend in backend_:
            backend = new_backend
        else:
            raise Exception("Unsupported backend ", new_backend)
    else:
        print("Using default backend (mkl)")
        backend = 'mkl'


def set_backend(new_backend):
    if not new_backend in backend_:
        raise Exception("Unsupported backend ", new_backend)
    else:
        global backend
        if new_backend != backend:
            print("Switched backed to ", new_backend)
            backend = new_backend


def get_backend():
    global backend
    if backend == None:
        __init__()
    return backend


