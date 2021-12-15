#! /usr/bin/env python3

import sys
import os
import pickle
import time
from ctypes import *
import random
import numpy as np
a, trash = os.path.split(os.path.abspath(__file__))
a, trash = os.path.split(os.path.abspath(a))
#path = os.path.join(a, 'matrix_vector_array_python')
sys.path.append(a)
print(a)
import backend
import libpdefd_vector_array as vector

print(backend.get_backend())


def random_list(N):
    l = []
    for _ in range(N):
        l.append(random.random())
    return l

N = 16000000

mkl = cdll.LoadLibrary("libmkl_rt.so")

#x = random_list(N)
#y = random_list(N)
x = np.random.rand(N)
y = np.random.rand(N)

x_vec = vector.vector_array(x)
y_vec = vector.vector_array(y)


#ix_c = (c_double*N)()
#y_c = (c_double*N)()

#for i in range(N):
#    x_c[i] = x[i]
#    y_c[i] = y[i]

#x_ptr = cast(x_c, POINTER(c_double)) #x.ctypes.data_as(POINTER(c_double))
#y_ptr = cast(y_c, POINTER(c_double)) #y.ctypes.data_as(POINTER(c_double))
print("setup finished")




k = 1
t = 0

for i in range(1,10,1):
    print(i, " threads")
    mkl.mkl_set_num_threads(byref(c_int(i)))
    print(mkl.mkl_get_max_threads())
    while t < 3:
        k *= 2
        print(k)
        start = time.time()
        for _ in range(k):
            x_vec += y_vec
            #mkl.cblas_daxpy(c_int(16000000), c_double(1), y_ptr, c_int(1), x_ptr, c_int(1))
        #mkl.vdAdd(16000000, x_ptr, y_ptr, x_ptr)
        end = time.time()
        t = end-start

    print(t/k)
    t = 0
    k = 1
