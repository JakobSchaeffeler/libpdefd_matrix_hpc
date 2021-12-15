#! /usr/bin/env python3

import sys
import os
import pickle
import time
import importlib
import numpy as np
import math
import scipy.sparse as sparse

from ctypes import *
a, trash = os.path.split(os.path.abspath(__file__))
a, trash = os.path.split(a)
path = os.path.join(a, 'matrix_vector_array')
sys.path.append(path)
import libpdefd_vector_array
import libpdefd_matrix_compute
import backend
from memory_profiler import profile

from ctypes import *

"""
Parameters for benchmarks
"""
storage = "data_matrix/"
max_cores = 24
N = 80000
N_squared = N * N
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

prefix = "_matrix"
print(backend.get_backend())
label = version + "_" + operation + "_" + str(N) + "_" + str(max_cores) + prefix 
label = "./" + storage + label

outfile = open(label, 'wb')
outfile_min = open(label + "_min", 'wb')
outfile_max = open(label + "_max", 'wb')

t_ = []
t_max_ = []
t_min_ = []
op = None
matr = None
arg1 = None
arg2 = None
arg3 = None

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
a = range(0,int(N*0.0001)+1)
diag = np.ones(N)
elem_mat_1 = []
for i in range(len(a)):
    elem_mat_1.append(diag * np.random.rand())
    
mat1 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))
m_diag1 = sparse.csr_matrix(mat1)
#m_diag1 = sp.random(N, N, density=0.00002, format='csr', dtype=float)
#matr = libpdefd_matrix_compute.matrix_sparse(sparse.csr_matrix(sparse.random(N, N, density=0.1)))
matr = libpdefd_matrix_compute.matrix_sparse(m_diag1)


if operation == 'iadd':
    op = libpdefd_matrix_compute.matrix_sparse.__iadd__
    a = range(0,int(N*0.0001)+1)
    diag = np.ones(N)
    elem_mat_1 = []
    for i in range(len(a)):
        elem_mat_1.append(diag * np.random.rand())
    mat1 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))
    m_diag1 = sparse.csr_matrix(mat1)
    #m_diag1 = sp.random(N, N, density=0.00002, format='csr', dtype=float)

    arg1 = libpdefd_matrix_compute.matrix_sparse(m_diag1)
elif operation == 'add':
    if version != 'scipy':
        print("no out of place add supported")
        op = libpdefd_matrix_compute.matrix_sparse.__iadd__
    else:    
        op = libpdefd_matrix_compute.matrix_sparse.__add__
    a = range(0,int(N*0.0001)+1)
    diag = np.ones(N)
    elem_mat_1 = []
    for i in range(len(a)):
        elem_mat_1.append(diag * np.random.rand())
    
    mat1 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))
    m_diag1 = sp.csr_matrix(mat1)
    arg1 = libpdefd_matrix_compute.matrix_sparse(m_diag1)


elif operation == 'dot_add_reshape':
    op = libpdefd_matrix_compute.matrix_sparse.dot_add_reshape
    arg1 = libpdefd_vector_array.vector_array(np.random.rand((N)))
    arg2 = libpdefd_vector_array.vector_array(np.random.rand((N)))
    arg3 = (2,int(N/2))



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
if operation == 'iadd' or operation == 'add':
    for _ in range(10):
        start = time.time()
        op(matr, arg1)
        end = time.time()
    k = 0
    t_overall = 0
    t_min = sys.maxsize
    t_max = 0

    while t_overall < min_time:
        k += 1
        start = time.time()
        op(matr, arg1)
        end = time.time()
        t = end - start
        if t < t_min:
            t_min = t
        elif t > t_max:
            t_max = t
        t_overall += t
        #print(t)
elif operation == 'dot_add_reshape':
    for _ in range(10):
        start = time.time()
        op(matr, arg1, arg2, arg3)
        end = time.time()
    k = 0
    t_overall = 0
    t_min = sys.maxsize
    t_max = 0

    while t_overall < min_time:
        k += 1
        start = time.time()
        op(matr, arg1, arg2, arg3)
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


