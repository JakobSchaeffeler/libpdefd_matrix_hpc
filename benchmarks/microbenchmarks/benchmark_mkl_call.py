#! /usr/bin/env python3

import sys
import os
import pickle
import time
import importlib
import numpy as np
import math
from ctypes import *


from ctypes import *

"""
Parameters for benchmarks
"""
max_cores = 24
N = 80000
"""minimum time of benchmark per operation"""
min_time = 5

mkl = cdll.LoadLibrary("libmkl_rt.so")
storage = "data/"
label = "data_mkl_call_py" 
label = "./" + storage + label

outfile = open(label, 'wb')
outfile_min = open(label + "_min", 'wb')
outfile_max = open(label + "_max", 'wb')

t_ = []
t_max_ = []
t_min_ = []
op = None
x = None
y = None
"""
Set cores
"""
#mkl.mkl_set_num_threads(byref(c_int(max_cores)))
#if version == 'mkl_blas_cython_cpp':
#    libpdefd_vector_array.set_max_num_threads(max_cores)         
#importlib.reload(libpdefd_vector_array)
op = mkl.vdAdd
x_tmp = np.random.rand(N)
x_np = np.empty(N)
x = x_np.ctypes.data_as(POINTER(c_double))
mkl.cblas_dcopy(c_int(N), x_tmp.ctypes.data_as(POINTER(c_double)), 1, x, 1)
y_tmp = np.random.rand(N)
y_np = np.empty(N)
y = y_np.ctypes.data_as(POINTER(c_double))
mkl.cblas_dcopy(c_int(N), y_tmp.ctypes.data_as(POINTER(c_double)), 1, y, 1)
"""
Setup
"""
print("Starting setup")
k = 0
t_overall = 0
t_min = sys.maxsize
t_max = 0


print("Setup finished")
"""
Benchmark
"""
k = 0
t_overall = 0
t_min = sys.maxsize
t_max = 0

"""
Warmup
"""
for _ in range(10):
    start = time.time()
    op(c_int(N), x,y,x)
    end = time.time()
k = 0
t_overall = 0
t_min = sys.maxsize
t_max = 0

while t_overall < min_time:
    k += 1
    start = time.time()
    op(c_int(N), x, y, x)
    end = time.time()
    t = end - start
    if t < t_min:
        t_min = t
    elif t > t_max:
        t_max = t
    t_overall += t

print("Benchmark (vdAdd) with",  max_cores, "cores finished, took ", str(t_overall/k), " seconds per operation (", N,"), min:", t_min, "; max:", t_max)


print("writing times")
pickle.dump(t_overall/k, outfile)
pickle.dump(t_max, outfile_max)
pickle.dump(t_min, outfile_min)
outfile.close()
outfile_max.close()
outfile_min.close()

