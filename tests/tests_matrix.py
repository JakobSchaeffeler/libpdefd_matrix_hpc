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

from ctypes import *

"""
Parameters for benchmarks
"""
max_cores = 24
N = 80000
N_squared = N * N
version = "scipy"
operation = "iadd"
"""maximum number of cores (step_width = 1)"""
if len(sys.argv) != 4:
    print("usage: python benchmark_vector max_cores N version op")

if len(sys.argv) == 4:
    N = int(sys.argv[1])
    version = sys.argv[2]
    operation = sys.argv[3]

x = np.random.rand(N,)
y = np.random.rand(N,)

print(operation)
print(version)
"""
Setup
"""
a = range(0,int(N*0.0001)+1)
diag = np.ones(N)
elem_mat_1 = []
for i in range(len(a)):
    elem_mat_1.append(diag * np.random.rand())
    
mat1 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))
m_diag1 = sparse.csr_matrix(mat1)
m_diag2 = None
matr = libpdefd_matrix_compute.matrix_sparse(m_diag1)
res1 = None

if operation == 'iadd':
    op = libpdefd_matrix_compute.matrix_sparse.__iadd__
    a = range(0,int(N*0.0001)+1)
    diag = np.ones(N)
    elem_mat_1 = []
    for i in range(len(a)):
        elem_mat_1.append(diag * np.random.rand())
    mat1 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))
    m_diag2 = sparse.csr_matrix(mat1)

    arg1 = libpdefd_matrix_compute.matrix_sparse(m_diag2)
    
    op(matr,arg1)
    res1 = matr.to_numpy_array()


elif operation == 'dot_add_reshape':
    op = libpdefd_matrix_compute.matrix_sparse.dot_add_reshape
    arg1 = libpdefd_vector_array.vector_array(x)
    arg2 = libpdefd_vector_array.vector_array(y)
    arg3 = (2,int(N/2))
    res_vec = op(matr, arg1, arg2, arg3)
    #print(res_vec.to_numpy_array())
    res1 = res_vec.to_numpy_array()
    
backend.set_backend('scipy')
importlib.reload(libpdefd_vector_array)
importlib.reload(libpdefd_matrix_compute)
res_sp = None
matr = libpdefd_matrix_compute.matrix_sparse(m_diag1)

if operation == 'iadd':
    op = libpdefd_matrix_compute.matrix_sparse.__iadd__
    arg1 = libpdefd_matrix_compute.matrix_sparse(m_diag2)
    
    op(matr,arg1)
    res_sp = matr.to_numpy_array()


elif operation == 'dot_add_reshape':
    op = libpdefd_matrix_compute.matrix_sparse.dot_add_reshape
    arg1 = libpdefd_vector_array.vector_array(x)
    arg2 = libpdefd_vector_array.vector_array(y)
    arg3 = (2,int(N/2))
    res_sp = op(matr, arg1, arg2, arg3).to_numpy_array()
    
print(res1)
print(res_sp)
assert np.isclose(res_sp - res1, 0).all()



