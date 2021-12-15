import cython
import numpy as np
cimport numpy as np
from ctypes import *
import sys
import os
import time
from libc.stdlib cimport malloc, free


cdef extern from "mkl.h":
    cdef void* mkl_malloc(size_t length, int alignment) nogil
    cdef void mkl_free(void* ptr) nogil




def run_mkl_malloc():
    return run_mkl_malloc_c()    
    
cdef run_mkl_malloc_c():
    t = 0
    cdef double* ptr
    for _ in range(1000000):
        start = time.time()
        ptr = <double*> mkl_malloc(128000000 * sizeof(double), 512)
        end = time.time()
        t += end-start
        mkl_free(ptr)
    return t

def run_np_malloc():
    return run_np_malloc_c()    
    
cdef run_np_malloc_c():
    t = 0
    cdef np.ndarray[double, ndim=1] res 
    cdef double* res_ptr
    for _ in range(1000000):
        start = time.time()
        res = np.empty(128000000)
        res_ptr = &res[0]
        end = time.time()
        t += end-start
    return t

       
