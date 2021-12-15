#! /usr/bin/env python3

import sys
import os
import pickle
import time
import importlib
import numpy as np
import math

a, trash = os.path.split(os.path.abspath(__file__))
a, trash = os.path.split(a)
a, trash = os.path.split(a)
path = os.path.join(a, 'matrix_vector_array')
sys.path.append(path)
import backend
import libpdefd_vector_array


from ctypes import *

"""
Parameters for benchmarks
"""
max_cores = 24
N = 80000
operation = "numpy_empty"
"""maximum number of cores (step_width = 1)"""
if len(sys.argv) != 4:
    print("usage: python benchmark_vector max_cores N version op")

if len(sys.argv) == 4:
    max_cores = int(sys.argv[1])
    N = int(sys.argv[2])
    operation = sys.argv[3]

"""minimum time of benchmark per operation"""
min_time = 5

mkl = cdll.LoadLibrary("libmkl_rt.so")
storage = "data/"
label = operation + "_" + str(N) + "_" + str(max_cores) 
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

"""
Setup
"""
print("Starting setup")
k = 0
t_overall = 0
t_min = sys.maxsize
t_max = 0
args = None
tmp = None
if operation == 'numpy_empty':
    op = np.empty
    args = N
elif operation == 'data_as':
    args = np.empty(N)
elif operation == 'init':
    tmp = np.random.rand(N)
    args = tmp.ctypes.data_as(POINTER(c_double)) 
    op = libpdefd_vector_array.vector_array_base.__init__

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
if operation == 'data_as':
    for _ in range(10):
        start = time.time()
        args.ctypes.data_as(POINTER(c_double))        
        end = time.time()
    k = 0
    t_overall = 0
    t_min = sys.maxsize
    t_max = 0

    while t_overall < min_time:
        k += 1
        start = time.time()
        args.ctypes.data_as(POINTER(c_double))
        end = time.time()
        t = end - start
        if t < t_min:
            t_min = t
        elif t > t_max:
            t_max = t
        t_overall += t

else:
    for _ in range(10):
        start = time.time()
        op(args)
        end = time.time()
    k = 0
    t_overall = 0
    t_min = sys.maxsize
    t_max = 0

    while t_overall < min_time:
        k += 1
        start = time.time()
        op(args)
        end = time.time()
        t = end - start
        if t < t_min:
            t_min = t
        elif t > t_max:
            t_max = t
        t_overall += t

print("Benchmark (", operation, ") with",  max_cores, "cores finished, took ", str(t_overall/k), " seconds per operation (", N,"), min:", t_min, "; max:", t_max)


print("writing times")
pickle.dump(t_overall/k, outfile)
pickle.dump(t_max, outfile_max)
pickle.dump(t_min, outfile_min)
outfile.close()
outfile_max.close()
outfile_min.close()

