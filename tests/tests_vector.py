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
max_cores = 24
N = 80000
version = "scipy"
operation = "iadd"
"""maximum number of cores (step_width = 1)"""
if len(sys.argv) != 4:
    print("usage: python benchmark_vector max_cores N version op")

if len(sys.argv) == 4:
    N = int(sys.argv[1])
    version = sys.argv[2]
    operation = sys.argv[3]

"""minimum time of benchmark per operation"""

mkl = cdll.LoadLibrary("libmkl_rt.so")

#importlib.reload(libpdefd_vector_array)
print(operation)
op = None

"""
Setup
"""
x = np.random.rand(N,)
y = np.random.rand(N,)
x_v = libpdefd_vector_array.vector_array(x)
y_v = libpdefd_vector_array.vector_array(y)


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
Tests
"""

one_ops = ["neg", "pos", "abs", "reduce_min", "create", "reduce_minabs", "reduce_max", "reduce_maxabs"]
scalar_ops = ["reduce_min", "reduce_min", "reduce_minabs", "reduce_max", "reduce_maxabs"] 

"""
Warmup
"""
res = None
res1 = None
if operation not in one_ops:
    res = op(x_v,y_v)
elif operation in one_ops:
    res = op(x_v)

if operation in scalar_ops:
    res1 = res
else:
    res1 = res.to_numpy_array()

backend.set_backend('scipy')
importlib.reload(libpdefd_vector_array)


x_np = libpdefd_vector_array.vector_array(x)
y_np = libpdefd_vector_array.vector_array(y)


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
Tests
"""

one_ops = ["neg", "pos", "abs", "reduce_min", "create", "reduce_minabs", "reduce_max", "reduce_maxabs", "reduce_min_omp"]

"""
Warmup
"""
res_np = None

if operation not in one_ops:
    res_np = op(x_np,y_np)
elif operation in one_ops:
    res_np = op(x_np)

if operation in scalar_ops:
    res_np = res
else:
    res_np = res.to_numpy_array()

if operation in scalar_ops:
    assert res_np == res1
else:
    assert np.isclose(res_np - res1, 0).all()

