#! /usr/bin/env python3

import sys
import os
import pickle
import time
import importlib
import numpy as np
import math
#import wrf
from ctypes import *
a, trash = os.path.split(os.path.abspath(__file__))
a, trash = os.path.split(a)
path = os.path.join(a, 'matrix_vector_array')
sys.path.append(path)
import backend
import libpdefd_vector_array


from memory_profiler import profile

from ctypes import *

"""
Parameters for benchmarks
"""
storage = "data_2911/"
max_cores = 24
N = 80000
version = "scipy"
operation = "iadd"
"""maximum number of cores (step_width = 1)"""
if len(sys.argv) != 5:
    print("usage: python benchmark_vector max_cores N version op")

if len(sys.argv) == 5:
    max_cores = int(sys.argv[1])
    N = int(sys.argv[2])
    version = sys.argv[3]
    operation = sys.argv[4]

"""minimum time of benchmark per operation"""
min_time = 5

mkl = cdll.LoadLibrary("libmkl_rt.so")

prefix = "_vector"
print(backend.get_backend())
#importlib.reload(libpdefd_vector_array)
label = version + "_" + operation + "_" + str(N) + "_" + str(max_cores) + prefix 
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

"""
eventuell loop einfügen hier über benchmark
"""
x = libpdefd_vector_array.vector_array(np.random.rand(N,))
y = libpdefd_vector_array.vector_array(np.random.rand(N,))


if operation == 'add':
    op = libpdefd_vector_array.vector_array_base.__add__
elif operation == 'iadd':
    op = libpdefd_vector_array.vector_array_base.__iadd__
elif operation == 'radd':
    op = libpdefd_vector_array.vector_array_base.__radd__
elif operation == 'sub':
    op = libpdefd_vector_array.vector_array_base.__sub__
elif operation == 'rsub':
    op = libpdefd_vector_array.vector_array_base.__rsub__
elif operation == 'isub':
    op = libpdefd_vector_array.vector_array_base.__isub__
elif operation == 'mul':
    op = libpdefd_vector_array.vector_array_base.__mul__
elif operation == 'imul':
    op = libpdefd_vector_array.vector_array_base.__imul__
elif operation == 'rmul':
    op = libpdefd_vector_array.vector_array_base.__rmul__
elif operation == 'truediv':
    op = libpdefd_vector_array.vector_array_base.__truediv__
elif operation == 'rtruediv':
    op =  libpdefd_vector_array.vector_array_base.__rtruediv__
elif operation == 'itruediv':
    op =  libpdefd_vector_array.vector_array_base.__itruediv__
elif operation == 'kron_vector':
    op =  libpdefd_vector_array.vector_array_base.kron_vector
elif operation == 'pow':
    op =  libpdefd_vector_array.vector_array_base.__pow__
elif operation == 'pos':
    op =  libpdefd_vector_array.vector_array_base.__pos__
elif operation == 'neg':
    op =  libpdefd_vector_array.vector_array_base.__neg__
elif operation == 'abs':
    op =  libpdefd_vector_array.vector_array_base.abs
elif operation == 'reduce_min':
    op =  libpdefd_vector_array.vector_array_base.reduce_min
elif operation == 'reduce_min_omp':
    op =  libpdefd_vector_array.vector_array_base.reduce_min_omp

elif operation == 'reduce_minabs':
    op =  libpdefd_vector_array.vector_array_base.reduce_minabs
elif operation == 'reduce_max':
    op =  libpdefd_vector_array.vector_array_base.reduce_min
elif operation == 'reduce_maxabs':
    op =  libpdefd_vector_array.vector_array_base.reduce_maxabs

elif operation == 'create':
    op = libpdefd_vector_array.vector_array
    x = np.random.rand(N,)



print("Setup finished")
"""
Benchmark
"""
k = 0
t_overall = 0
t_min = sys.maxsize
t_max = 0

one_ops = ["neg", "pos", "abs", "reduce_min", "create", "reduce_minabs", "reduce_max", "reduce_maxabs", "reduce_min_omp"]

"""
Warmup
"""
if operation not in one_ops:
    for _ in range(10):
        start = time.time()
        op(x,y)
        end = time.time()
    k = 0
    t_overall = 0
    t_min = sys.maxsize
    t_max = 0

    while t_overall < min_time:
        k += 1
        start = time.time()
        op(x,y)
        end = time.time()
        t = end - start
        if t < t_min:
            t_min = t
        elif t > t_max:
            t_max = t
        t_overall += t
elif operation in one_ops:
    for _ in range(10):
        start = time.time()
        op(x)
        end = time.time()
    k = 0
    t_overall = 0
    t_min = sys.maxsize
    t_max = 0

    while t_overall < min_time:
        k += 1
        start = time.time()
        op(x)
        end = time.time()
        t = end - start
        if t < t_min:
            t_min = t
        elif t > t_max:
            t_max = t
        t_overall += t
else:
    raise Exception("operation not supported by tests")

print("Benchmark (", operation, " ", version ,") with",  max_cores, "cores finished, took ", str(t_overall/k), " seconds per operation (", N,"), min:", t_min, "; max:", t_max)


print("writing times")
pickle.dump(t_overall/k, outfile)
pickle.dump(t_max, outfile_max)
pickle.dump(t_min, outfile_min)
outfile.close()
outfile_max.close()
outfile_min.close()

"""
Benchmarks scalars
"""
if operation not in one_ops and operation != 'kron_vector':
     
    prefix = "_scalar"
    #backend.set_backend(version)
    #importlib.reload(libpdefd_vector_array)
    label = version + "_" + operation + "_" + str(N) + "_" + str(max_cores) + prefix 
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

           
    """
    eventuell loop einfügen hier über benchmark
    """
    x = libpdefd_vector_array.vector_array(np.random.rand(N,))
    y = np.random.rand()

    if operation == 'add':
        op = libpdefd_vector_array.vector_array_base.__add__
    elif operation == 'iadd':
        op = libpdefd_vector_array.vector_array_base.__iadd__
    elif operation == 'radd':
        op = libpdefd_vector_array.vector_array_base.__radd__
    elif operation == 'sub':
        op = libpdefd_vector_array.vector_array_base.__sub__
    elif operation == 'rsub':
        op = libpdefd_vector_array.vector_array_base.__rsub__
    elif operation == 'isub':
        op = libpdefd_vector_array.vector_array_base.__isub__
    elif operation == 'mul':
        op = libpdefd_vector_array.vector_array_base.__mul__
    elif operation == 'imul':
        op = libpdefd_vector_array.vector_array_base.__imul__
    elif operation == 'rmul':
        op = libpdefd_vector_array.vector_array_base.__rmul__
    elif operation == 'truediv':
        op = libpdefd_vector_array.vector_array_base.__truediv__
    elif operation == 'rtruediv':
        op =  libpdefd_vector_array.vector_array_base.__rtruediv__
    elif operation == 'itruediv':
        op =  libpdefd_vector_array.vector_array_base.__itruediv__
    elif operation == 'kron_vector':
        op =  libpdefd_vector_array.vector_array_base.kron_vector
    elif operation == 'pow':
        op =  libpdefd_vector_array.vector_array_base.__pow__


    print("Setup finished")
    """
    Benchmark
    """

    """
    Warmup
    """
    for _ in range(10):
        start = time.time()
        op(x,y)
        end = time.time()
    k = 0
    t_overall = 0
    t_min = sys.maxsize
    t_max = 0

    while t_overall < min_time:
        k += 1
        start = time.time()
        op(x,y)
        end = time.time()
        t = end - start
        if t < t_min:
            t_min = t
        elif t > t_max:
            t_max = t
        t_overall += t

    print("Benchmark (", operation, " ", version ,") with",  max_cores, "cores finished, took ", str(t_overall/k), " seconds per operation (", N,"), min:", t_min, "; max:", t_max)


    print("writing times")
    pickle.dump(t_overall/k, outfile)
    pickle.dump(t_max, outfile_max)
    pickle.dump(t_min, outfile_min)
    outfile.close()
    outfile_max.close()
    outfile_min.close()


