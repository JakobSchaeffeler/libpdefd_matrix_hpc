#! /usr/bin/env python3

from ctypes import *
mkl = cdll.LoadLibrary("libmkl_rt.so")


# print(mkl.mkl_disable_fast_mm())
import sys
import os
import pickle
import time
import importlib
import numpy as np
import math
#import wrf
import tracemalloc


a, trash = os.path.split(os.path.abspath(__file__))
a, trash = os.path.split(a)
path = os.path.join(a, 'matrix_vector_array')
sys.path.append(path)
import backend
import libpdefd_vector_array

import python_loop
#import cython_loop

from ctypes import *

prefix = ""
if len(sys.argv) == 2:
    prefix = "_" + str(sys.argv[1])

# print(os.environ['KMP_AFFINITY'])

"""
Parameters for benchmarks
"""

"""maximum number of cores (step_width = 1)"""
max_cores = 4

"""minimum time of benchmark per operation"""
min_time = 5

"""Length of Vectors"""
#N_ = (80000,)
N_ = (80000000,)
#N_ = (268435456,)
"""Tested operations"""
#op_ = ('sub',  'isub', 'truediv', 'rtruediv', 'itruediv', 'pow')#('add', 'iadd', 'mul', 'imul', 'rmul','rsub','radd', )

op_ = ('iadd',)


op_vec_ = ('kron_vector',)
"""tested python implementations"""
version_ =  'scipy', 'mkl', 'mkl_blas'
version_ = ()
"""tested cython implementations"""
#version_cython_ = 'scipy_cython', 'mkl_cython_naiv', 'mkl_cython_typed', 'mkl_cython_typed_cdef', 'mkl_blas_cython_naiv', 'mkl_blas_cython_typed', 'mkl_blas_cython_typed_cdef'

#version_cython_ = 'mkl_cython_c', 'mkl_blas_cython_c'

version_cython_ = ('mkl_blas_cython_cpp', )

benchmark_config_ = []
print(os.environ)
"""
Benchmark configs for experimental versions
"""
backend_ = ['scipy', 'scipy_c', 'mkl', 'mkl_cython', 'mkl_cython_c', 'mkl_blas', 'mkl_cython_malloc', 'mkl_blas_cython', 'mkl_blas_cython_c']


# for N in N_:
#     benchmark_config = {'outer_loop': 'cython', 'inner_loop': 'cython', 'N': N, 'op': 'add',
#                         'version': 'mkl_cython_c'}
#     benchmark_config_ += [benchmark_config]
#
# for N in N_:
#     benchmark_config = {'outer_loop': 'python', 'inner_loop': 'cython', 'N': N, 'op': 'add',
#                         'version': 'mkl_cython_c'}
#     benchmark_config_ += [benchmark_config]
#
# for N in N_:
#     benchmark_config = {'outer_loop': 'python', 'inner_loop': 'cython', 'N': N, 'op': 'add',
#                         'version': 'mkl_cython_malloc'}
#     benchmark_config_ += [benchmark_config]
#
# for N in N_:
#     benchmark_config = {'outer_loop': 'cython', 'inner_loop': 'cython', 'N': N, 'op': 'add',
#                         'version': 'mkl_cython_malloc'}
#     benchmark_config_ += [benchmark_config]

#for N in N_:
#    benchmark_config = {'outer_loop': 'python', 'inner_loop': 'cython', 'N': N, 'op': 'add_fun',
#                            'version': 'mkl_c'}
#    benchmark_config_ += [benchmark_config]


#for N in N_:
#    benchmark_config = {'outer_loop': 'cython', 'inner_loop': 'cython', 'N': N, 'op': 'add_fun',
#                            'version': 'mkl'}
#    benchmark_config_ += [benchmark_config]



"""
Configs for main versions
"""
# print(os.environ['KMP_AFFINITY'])
for N in N_:
    for op in op_:
        for version in version_:
            benchmark_config = {'outer_loop': 'python', 'inner_loop': 'python', 'N': N, 'op': op,
                                'version': version}
            benchmark_config_ += [benchmark_config]

for op in op_:
    for N in N_:
        for version in version_cython_:
            benchmark_config = {'outer_loop': 'python', 'inner_loop': 'cython', 'N': N, 'op': op,
                                 'version': version}
            benchmark_config_ += [benchmark_config]

#for N in N_:
#    for version in version_cython_:
#        benchmark_config = {'outer_loop': 'cython', 'inner_loop': 'cython', 'N': N, 'op': 'add',
#                                'version': version}
#        benchmark_config_ += [benchmark_config]




#os.environ['KMP_BLOCKTIME'] = "30"
#os.environ['OMP_PLACES'] = 'cores'
#os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
#os.environ['KMP_SETTINGS'] = "1"

num_config = len(benchmark_config_) * 2
iteration = 0

"""
Benchmarks vectors
"""

#for N in N_:
#    for op in op_vec_:
#        for version in version_:
#            benchmark_config = {'outer_loop': 'python', 'inner_loop': 'python', 'N': int(math.sqrt(N)), 'op': op,
#                                'version': version}
#            benchmark_config_ += [benchmark_config]

prefix = ""
for config in benchmark_config_:
    #print(config['version'])
    #backend.set_backend(config['version'])
    #importlib.reload(libpdefd_vector_array)
    for i in (24, 24, 24):
        #libpdefd_vector_array.set_max_num_threads(i)
        #importlib.reload(libpdefd_vector_array)
        #libpdefd_vector_array.set_max_num_threads(i)
        #tracemalloc.start()
        #libpdefd_vector_array.test_iadd_mkl_realloc()   
        #snapshot1 = tracemalloc.take_snapshot()
        #print("np")
        #libpdefd_vector_array.test_iadd_np_empty(N,i)
        #snapshot2 = tracemalloc.take_snapshot()
        print("alloc")
        #libpdefd_vector_array.test_empty(N)
        libpdefd_vector_array.test_iadd_alloc(N,i)
        #print("mkl_malloc")
        #libpdefd_vector_array.test_iadd_mkl_malloc(N,i)
        #print("malloc")
        #libpdefd_vector_array.test_iadd_malloc(N,i)
        #print("numa")
        #libpdefd_vector_array.test_iadd_numa_alloc(N,i)


