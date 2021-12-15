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

import python_loop
#import cython_loop

from ctypes import *

"""
Parameters for benchmarks
"""

max_cores = 24

"""maximum number of cores (step_width = 1)"""
if len(sys.argv) != 2:
    print("No maximum number of cores specified, using 24!")

if len(sys.argv) == 2:
    max_cores = int(sys.argv[1])



"""minimum time of benchmark per operation"""
min_time = 5

"""Length of Vectors"""
N_ = (134217728,)
N_ = (80000,)
"""Tested operations"""
#op_ = ('sub',  'isub', 'truediv', 'rtruediv', 'itruediv', 'pow')#('add', 'iadd', 'mul', 'imul', 'rmul','rsub','radd', )

op_ = ('iadd', 'add', )


op_vec_ = ('kron_vector',)
"""tested python implementations"""
#version_ =  'scipy', 'mkl', 'mkl_blas'
version_ = ('mkl_blas',)
#version_=()
"""tested cython implementations"""
#version_cython_ = 'scipy_cython', 'mkl_cython_naiv', 'mkl_cython_typed', 'mkl_cython_typed_cdef', 'mkl_blas_cython_naiv', 'mkl_blas_cython_typed', 'mkl_blas_cython_typed_cdef'

#version_cython_ = 'mkl_cython_c', 'mkl_blas_cython_c'

version_cython_ = ('mkl_blas_cython_cpp', 'mkl_blas_cython_c' )
#version_cython_ = ()
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
#print(os.environ['KMP_AFFINITY'])
for op in op_:
    for N in N_:
        for version in version_cython_:
            benchmark_config = {'outer_loop': 'python', 'inner_loop': 'cython', 'N': N, 'op': op,
                                 'version': version}
            benchmark_config_ += [benchmark_config]
for N in N_:
    for op in op_:
        for version in version_:
            benchmark_config = {'outer_loop': 'python', 'inner_loop': 'python', 'N': N, 'op': op,
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

mkl = cdll.LoadLibrary("libmkl_rt.so")
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
    print(config['version'])
    backend.set_backend(config['version'])
    importlib.reload(libpdefd_vector_array)

    label = "outer_" + config.get('outer_loop') + "_inner_" + config.get('inner_loop') + "_" + config['version'] +\
            "_" + str(config['N']) + "_" + config['op'] 
    label = "./data_2411_sk1/" + label
    outfile = open(label, 'wb')
    outfile_min = open(label + "_min", 'wb')
    outfile_max = open(label + "_max", 'wb')

    t_ = []
    t_max_ = []
    t_min_ = []
    op = None
    x = None
    y = None
    for i in range(max_cores, max_cores+1, 1):
        print("staring for ", i, " cores")
        """
        Set cores
        """
        mkl.mkl_set_num_threads(byref(c_int(i)))
        print(mkl.mkl_get_max_threads())
        if config['version'] == 'mkl_blas_cython_cpp':
            libpdefd_vector_array.set_max_num_threads(i)         
        importlib.reload(libpdefd_vector_array)
        print("set mkl threads")
        #print(wrf.omp_get_max_threads())
        """
        Setup
        """
        print("Starting setup")
        k = 0
        t_overall = 0
        t_min = sys.maxsize
        t_max = 0

       
        for _ in range(1,3):
            x = libpdefd_vector_array.vector_array(np.random.rand(config['N'],))
            y = libpdefd_vector_array.vector_array(np.random.rand(config['N'],))
            #x = libpdefd_vector_array.vector_array(np.random.uniform(low=0.0, high=1.0, size=(config['N'],)))
            #y = libpdefd_vector_array.vector_array(np.random.uniform(low=0.0, high=1.0, size=(config['N'],)))
       
            if config['outer_loop'] == 'python':
                if config['op'] == 'add':
                    op = libpdefd_vector_array.vector_array_base.__add__
                elif config['op'] == 'iadd':
                    op = libpdefd_vector_array.vector_array_base.__iadd__
                elif config['op'] == 'radd':
                    op = libpdefd_vector_array.vector_array_base.__radd__
                elif config['op'] == 'sub':
                    op = libpdefd_vector_array.vector_array_base.__sub__
                elif config['op'] == 'rsub':
                    op = libpdefd_vector_array.vector_array_base.__rsub__
                elif config['op'] == 'isub':
                    op = libpdefd_vector_array.vector_array_base.__isub__
                elif config['op'] == 'mul':
                    op = libpdefd_vector_array.vector_array_base.__mul__
                elif config['op'] == 'imul':
                    op = libpdefd_vector_array.vector_array_base.__imul__
                elif config['op'] == 'rmul':
                    op = libpdefd_vector_array.vector_array_base.__rmul__
                elif config['op'] == 'truediv':
                    op = libpdefd_vector_array.vector_array_base.__truediv__
                elif config['op'] == 'rtruediv':
                    op =  libpdefd_vector_array.vector_array_base.__rtruediv__
                elif config['op'] == 'itruediv':
                    op =  libpdefd_vector_array.vector_array_base.__itruediv__
                elif config['op'] == 'kron_vector':
                    op =  libpdefd_vector_array.vector_array_base.kronvector
                elif config['op'] == 'pow':
                    op =  libpdefd_vector_array.vector_array_base.__pow__
                elif config['op'] == 'pos':
                    op =  libpdefd_vector_array.vector_array_base.__pos__



            elif config['inner_loop'] == 'cython' and config['outer_loop'] == 'cython':
                #op = cython_loop.add_vector
                op = python_loop.add_vector


            print("Setup finished")
            """
            Benchmark
            """

            """
            Warmup
            """
            if config['op'] == 'pos':
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
                    op(x, y)
                    end = time.time()
                    t = end - start
                    if t < t_min:
                        t_min = t
                    elif t > t_max:
                        t_max = t
                    t_overall += t
        t_.append(t_overall/k)
        t_max_.append(t_max)
        t_min_.append(t_min)
        print("Benchmark (", config['op'], " ", config['version'] ,") ",  i, "of iteration ", iteration, "finished, took ", t_overall/k, " seconds per operation (", config['N'],"), min:", t_min, "; max:", t_max)


    print("writing times")
    pickle.dump(t_, outfile)
    pickle.dump(t_max_, outfile_max)
    pickle.dump(t_min_, outfile_min)
    outfile.close()
    outfile_max.close()
    outfile_min.close()
    iteration += 1
    print(iteration/num_config * 100, "% done")



"""
Benchmarks scalars
"""
prefix = "_scalar"
for config in benchmark_config_:
    backend.set_backend(config['version'])
 
    importlib.reload(libpdefd_vector_array)
    print(backend.get_backend())
    print(config['op'])
    label = "outer_" + config.get('outer_loop') + "_inner_" + config.get('inner_loop') + "_" + config['version'] + \
            "_" + str(config['N']) + "_" + config['op'] + prefix
    label = "./data_2411_sk1/" + label
    outfile = open(label, 'wb')
    outfile_min = open(label + "_min", 'wb')
    outfile_max = open(label + "_max", 'wb')
    t_ = []
    t_max_ = []
    t_min_ = []
   
    
    op = None
    x = None
    y = None
    for i in range(max_cores, max_cores+1, 1):
        print("staring for ", i, " cores")
        """
        Set cores
        """
        print("set mkl threads")
        importlib.reload(libpdefd_vector_array)
        mkl.mkl_set_num_threads(byref(c_int(i)))
        """
        Setup
        """
        print("Starting setup")
        y = np.random.rand()
        x = libpdefd_vector_array.vector_array(np.random.rand(config['N'],))
        if config['outer_loop'] == 'python':
            if config['op'] == 'add':
                op = libpdefd_vector_array.vector_array_base.__add__
            elif config['op'] == 'iadd':
                op = libpdefd_vector_array.vector_array_base.__iadd__
            elif config['op'] == 'radd':
                op = libpdefd_vector_array.vector_array_base.__radd__
            elif config['op'] == 'sub':
                op = libpdefd_vector_array.vector_array_base.__sub__
            elif config['op'] == 'rsub':
                op = libpdefd_vector_array.vector_array_base.__rsub__
            elif config['op'] == 'isub':
                op = libpdefd_vector_array.vector_array_base.__isub__
            elif config['op'] == 'mul':
                op = libpdefd_vector_array.vector_array_base.__mul__
            elif config['op'] == 'imul':
                op = libpdefd_vector_array.vector_array_base.__imul__
            elif config['op'] == 'rmul':
                op = libpdefd_vector_array.vector_array_base.__rmul__
            elif config['op'] == 'truediv':
                op = libpdefd_vector_array.vector_array_base.__truediv__
            elif config['op'] == 'rtruediv':
                op =  libpdefd_vector_array.vector_array_base.__rtruediv__
            elif config['op'] == 'itruediv':
                op =  libpdefd_vector_array.vector_array_base.__itruediv__
            elif config['op'] == 'kron_vector':
                op =  libpdefd_vector_array.vector_array_base.kronvector
            elif config['op'] == 'pow':
                op =  libpdefd_vector_array.vector_array_base.__pow__
            elif config['op'] == 'pos':
                op =  libpdefd_vector_array.vector_array_base.__pos__



        elif config['inner_loop'] == 'cython' and config['outer_loop'] == 'cython':
             op = python_loop.add
 
 
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
            op(x, y)
            end = time.time()
            t = end - start
            if t < t_min:
                t_min = t
            elif t > t_max:
                t_max = t
            t_overall += t
        t_.append(t_overall/k)
        t_max_.append(t_max)
        t_min_.append(t_min)
        print("Benchmark (", config['op'], " ", config['version'] ,") ",  i, "of iteration ", iteration, "finished, took ", t_overall/k, " seconds per operation (", config['N'],"), min:", t_min, "; max:", t_max)


    print("writing times")
    pickle.dump(t_, outfile)
    pickle.dump(t_max_, outfile_max)
    pickle.dump(t_min_, outfile_min)
    outfile.close()
    outfile_max.close()
    outfile_min.close()
           
    iteration += 1
    print(iteration/num_config * 100, "% done")
 
 



