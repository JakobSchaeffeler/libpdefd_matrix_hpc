#! /usr/bin/env python3

import sys, os
import time
import numpy as np
import random
import scipy.sparse as sparse

"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.matrix_vector_array.libpdefd_matrix_compute_mkl as libpdefd_matrix_compute_mkl

except:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import libpdefd_matrix_compute_cython_mkl as libpdefd_matrix_compute_mkl
    sys.path.pop()
    sys.path.append(os.path.join(os.path.dirname(__file__), "mkl_cpp_alloc"))
    """
    backend has to handle version used
    """
    import libpdefd_vector_array_cython_mkl_cpp_alloc as libpdefd_vector_array

"""
Problem size
"""
N = 8000

if len(sys.argv) > 1:
    N = int(sys.argv[1])

"""
Number of iterations
"""
K = 1

if len(sys.argv) > 2:
    K = int(sys.argv[2])


print("")
print("*"*80)
print("Array A")
print("*"*80)
dim1 = random.randint(N,N+10)
dim2 = random.randint(N,N+10)

a = sparse.csr_matrix(sparse.random(dim1, dim2, density=0.25))
b = sparse.csr_matrix(sparse.random(dim1, dim2, density=0.25))
print("*"*80)
print("Matrix Compute")
print("*"*80)
m_compute = libpdefd_matrix_compute_mkl.matrix_sparse(a)
print("FIN")
m_compute2 = libpdefd_matrix_compute_mkl.matrix_sparse(b)


print("*"*80)
print("Benchmarks")
print("*"*80)
start = time.time()
if 1:
    print("add test")


    m_compute.__iadd__(m_compute2)

    a += b


    assert np.isclose(m_compute.to_numpy_array() - a, 0).all()


matrix_sp = sparse.csr_matrix(a)

a_vec = np.random.rand(dim2)
b_vec = np.random.rand(dim1)

vector = libpdefd_vector_array.vector_array(a_vec)
vector2 = libpdefd_vector_array.vector_array(b_vec)
if 1:
    print("dot_add test 1")
    retval2 = m_compute.dot_add_reshape(vector, vector2, b_vec.shape)

    res2 = retval2.to_numpy_array()
    sp_res = (matrix_sp.dot(a_vec.flatten()) + b_vec).reshape(b_vec.shape)
    assert retval2.shape == b_vec.shape
    assert np.isclose(retval2.to_numpy_array() - sp_res, 0).all()
end = time.time()


print("Benchmark took ",end-start, " seconds")
