#! /usr/bin/env python3

import sys
import os
import pickle
import time
import python_loop
import cython_loop
import importlib
import numpy as np
import scipy.sparse as sparse
a, trash = os.path.split(os.path.dirname(__file__))
path = os.path.join(a, 'matrix_vector_array')
sys.path.append(path)
print(path)
"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.matrix_vector_array.libpdefd_vector_array as libpdefd_vector_array
    import libpdefd.matrix_vector_array.libpdefd_matrix_compute as libpdefd_matrix_compute
    import libpdefd.matrix_vector_array.backend

except:
    print(path)
    cy_path = os.path.join(path, 'matrix_vector_array_cython')
    sys.path.append(cy_path)
    import libpdefd_matrix_compute_cython_mkl as libpdefd_matrix_compute_cython_mkl
    sys.path.append(os.path.join(cy_path, "mkl_blas_c"))
    import libpdefd_vector_array_cython_mkl_blas_c as libpdefd_vector_array_cython_mkl_blas_c
    sys.path.pop()
    sys.path.pop()
    sys.path.pop()
    path = os.path.join(a, 'matrix_vector_array')
    sys.path.append(path)
    py_path = os.path.join(path, 'matrix_vector_array_python')
    sys.path.append("../matrix_vector_array/matrix_vector_array_python")
    print(py_path)
    import matrix_vector_array_python.libpdefd_matrix_compute_mkl as libpdefd_matrix_compute_mkl
    import libpdefd_matrix_compute_sp as libpdefd_matrix_compute_sp
    import libpdefd_vector_array_mkl_blas as libpdefd_vector_array_mkl_blas
    import libpdefd_vector_array_np as libpdefd_vector_array_np
    sys.path.pop()



from ctypes import *

if len(sys.argv) > 2:
    raise Exception("Please add prefix for benchmark names")
prefix = ""
if len(sys.argv) == 2:
    prefix = "_" + str(sys.argv[1])


"""
Parameters for benchmarks
"""
max_cores = 10
N_ = (80000,)
version_ = "scipy", "mkl"
version_cython_ = ("mkl_cython_c",)
benchmark_config_ = []

"""
Benchmark configs for experimental versions
"""
backend_ = ['scipy', 'mkl', 'mkl_cython']

"""
Configs for main versions
"""

for N in N_:
    for version in version_:
        benchmark_config = {'outer_loop': 'python', 'inner_loop': 'python', 'N': N,
                            'version': version}
        benchmark_config_ += [benchmark_config]

for N in N_:
    for version in version_cython_:
        benchmark_config = {'outer_loop': 'python', 'inner_loop': 'cython', 'N': N,
                            'version': version}
        benchmark_config_ += [benchmark_config]


for N in N_:
    for version in version_cython_:
        benchmark_config = {'outer_loop': 'cython', 'inner_loop': 'cython', 'N': N,
                            'version': version}
        benchmark_config_ += [benchmark_config]






os.environ['KMP_BLOCKTIME'] = "30"
os.environ['OMP_PLACES'] = 'cores'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
os.environ['KMP_SETTINGS'] = "1"

mkl = cdll.LoadLibrary("libmkl_rt.so")
num_config = len(benchmark_config_) * 2
iteration = 0


"""
Benchmarks dot_add_reshape
"""
prefix = "_dot_add_reshape"
for config in benchmark_config_:
    label = "outer_" + config.get('outer_loop') + "_inner_" + config.get('inner_loop') + "_" + config['version'] + \
            "_" + str(config['N']) + prefix
    label = "./data_matrix/" + label
    outfile = open(label, 'wb')
    t_ = []
    op = None
    x = None
    y = None
    matrix = None
    for i in range(1, max_cores+1, 1):
        print("staring for ", i, " cores")
        """
        Set cores
        """
        print("set mkl threads")
        mkl.mkl_set_num_threads(byref(c_int(i)))
        print("set omp threads")
        os.environ['OMP_NUM_THREADS'] = str(i) #probably useless
        """
        Setup
        """
        print("Starting setup")
        N = config['N']
        a = range(-6,6)
        diag = np.ones(N)
        elem_mat_1 = []
        for i in range(len(a)):
            elem_mat_1.append(diag * np.random.rand())

        # general setup
        mat1 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))

        m_diag1 = sparse.csr_matrix(mat1)

        if config['inner_loop'] == 'python':

            if config['version'] == "scipy":
                matrix = libpdefd_matrix_compute_sp.matrix_sparse(m_diag1)
                x = libpdefd_vector_array_np.vector_array(np.random.rand(N))
                y = libpdefd_vector_array_np.vector_array(np.random.rand(N))
            elif config['version'] == 'mkl':
                matrix = libpdefd_matrix_compute_mkl.matrix_sparse(m_diag1)
                x = libpdefd_vector_array_mkl_blas.vector_array(np.random.rand(N))
                y = libpdefd_vector_array_mkl_blas.vector_array(np.random.rand(N))

            op = python_loop.dot_add_reshape

        elif config['inner_loop'] == 'cython' and config['outer_loop'] == 'cython':
            op = cython_loop.dot_add_reshape_cython
            matrix = libpdefd_matrix_compute_cython_mkl.matrix_sparse(m_diag1)
            x = libpdefd_vector_array_cython_mkl_blas_c.vector_array(np.random.rand(N))
            y = libpdefd_vector_array_cython_mkl_blas_c.vector_array(np.random.rand(N))

        else:
            op = python_loop.dot_add_reshape
            matrix = libpdefd_matrix_compute_cython_mkl.matrix_sparse(m_diag1)
            x = libpdefd_vector_array_cython_mkl_blas_c.vector_array(np.random.rand(N))
            y = libpdefd_vector_array_cython_mkl_blas_c.vector_array(np.random.rand(N))

        matrix.dot_add_reshape(x,y,(N,))
        dst_shape = (N,)
        print(dst_shape)
        print("Setup finished")
        """
        Benchmark
        """
        k = 1
        t = 0
        while t < 10:
            k *= 2
            start = time.time()
            op(matrix, x, y, dst_shape, k)
            end = time.time()
            t = end - start
        t_.append(t/k)
        print("Benchmark ", i, "of iteration ", iteration, "finished, took ", t/k, " seconds per operation")


    print("writing times")
    pickle.dump(t_, outfile)
    outfile.close()
    iteration += 1
    print(iteration/num_config * 100, "% done")


"""
Benchmarks iadd
"""
prefix = "_iadd"
for config in benchmark_config_:
    print(config['version'])

    label = "outer_" + config.get('outer_loop') + "_inner_" + config.get('inner_loop') + "_" + config['version'] + \
            "_" + str(config['N'])  + prefix
    label = "./data_matrix/" + label
    outfile = open(label, 'wb')
    t_ = []
    op = None
    for i in range(1, max_cores+1, 1):
        print("staring for ", i, " cores")
        """
        Set cores
        """
        print("set mkl threads")
        mkl.mkl_set_num_threads(byref(c_int(i)))
        print("set omp threads")
        os.environ['OMP_NUM_THREADS'] = str(i) #probably useless
        """
        Setup
        """
        print("Starting setup")
        N = config['N']
        a = range(-6,6)
        diag = np.ones(N)
        elem_mat_1 = []
        elem_mat_2 = []
        for i in range(len(a)):
            elem_mat_1.append(diag * np.random.rand())
            elem_mat_2.append(diag * np.random.rand())

        # general setup
        mat1 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))
        mat2 = sparse.dia_matrix((elem_mat_1, [i for i in a]), shape=(N,N))

        m_diag1 = sparse.csr_matrix(mat1)
        m_diag2 = sparse.csr_matrix(mat2)

        A = None
        B = None

        if config['inner_loop'] == 'python':
            op = python_loop.add_matrix

            if config['version'] == "scipy":
                A = libpdefd_matrix_compute_sp.matrix_sparse(m_diag1)
                B = libpdefd_matrix_compute_sp.matrix_sparse(m_diag2)
            elif config['version'] == 'mkl':
                A = libpdefd_matrix_compute_mkl.matrix_sparse(m_diag1)
                B = libpdefd_matrix_compute_mkl.matrix_sparse(m_diag2)

            op = python_loop.add_matrix

        elif config['inner_loop'] == 'cython' and config['outer_loop'] == 'cython':
            op = cython_loop.add_matrix
            A = libpdefd_matrix_compute_cython_mkl.matrix_sparse(m_diag1)
            B = libpdefd_matrix_compute_cython_mkl.matrix_sparse(m_diag2)

        else:
            op = python_loop.add_matrix
            A = libpdefd_matrix_compute_cython_mkl.matrix_sparse(m_diag1)
            B = libpdefd_matrix_compute_cython_mkl.matrix_sparse(m_diag2)

        print("Setup finished")
        """
        Benchmark
        """
        k = 1
        t = 0
        while t < 10:
            k *= 2
            start = time.time()
            op(A, B, k)
            end = time.time()
            t = end - start
        t_.append(t/k)
        print("Benchmark ", i, "of iteration ", iteration, "finished, took ", t/k, " seconds per operation")


    print("writing times")
    pickle.dump(t_, outfile)
    outfile.close()
    iteration += 1
    print(iteration/num_config * 100, "% done")


