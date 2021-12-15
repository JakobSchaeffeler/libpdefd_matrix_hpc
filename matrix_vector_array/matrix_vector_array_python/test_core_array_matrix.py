#! /usr/bin/env python3

import sys, os
import time
import numpy as np
import backend
backend.set_backend('mkl')


"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.matrix_vector_array.libpdefd_vector_array as libpdefd_vector_array
    import libpdefd.matrix_vector_array.libpdefd_matrix_setup as libpdefd_matrix_setup
    import libpdefd.matrix_vector_array.libpdefd_matrix_compute as libpdefd_matrix_compute

except:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import libpdefd_vector_array as libpdefd_vector_array
    import libpdefd_matrix_setup as libpdefd_matrix_setup
    import libpdefd_matrix_compute as libpdefd_matrix_compute
    sys.path.pop()



"""
Problem size
"""
N = 2

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
a = libpdefd_vector_array.vector_array_zeros((N,N))
al = np.prod(a.shape)



for z in range(a.shape[0]):
    for y in range(a.shape[1]):
        a[z,y] = y*a.shape[1] + z*(a.shape[1]*a.shape[0])



print("")
print("*"*80)
print("Array B")
print("*"*80)
b = libpdefd_vector_array.vector_array_zeros((N,N))
bl = np.prod(b.shape)

for z in range(b.shape[0]):
    for y in range(b.shape[1]):
        b[z,y] = y*b.shape[1] + z*(b.shape[1]*b.shape[0]) + 5
print(b)



print("")
print("*"*80)
print("Array C")
print("*"*80)
c = b*2.0
c = c.flatten()
print(c)



print("")
print("*"*80)
print("Matrix Sparse")
print("*"*80)

print(" + allocation")
m_setup = libpdefd_matrix_setup.matrix_sparse(shape=(bl, al))

print(" + setup")
for i in range(min(al, bl)):
    m_setup[i,i] = -2

for i in range(min(al, bl)-1):
    m_setup[i,i+1] = 1
    m_setup[i,i-1] = 3

print(m_setup)


print("*"*80)
print("Matrix Compute")
print("*"*80)

m_compute = libpdefd_matrix_compute.matrix_sparse(m_setup)
print("FIN")
print(m_compute.to_numpy_array())



print("*"*80)
print("Benchmarks")
print("*"*80)

time_start = time.time()
res1 = None
res2 = None
res3 = None
for k in range(K):
    print("Iteration: "+str(k))

    if 1:
        print("MUL test 1")
        retval1 = m_compute.dot__DEPRECATED(a.flatten())
        res1 = retval1.to_numpy_array()
        print(res1)
        assert retval1.shape == c.shape


    if 1:
        print("MUL test 2")
        retval2 = m_compute.dot_add_reshape(a, c, b.shape)
        res2 = retval2.to_numpy_array()
        print(res2)
        assert retval2.shape == b.shape


    if 1:
        print("MUL test 3")
        retval3 = m_compute.dot_add_reshape(a, c, b.shape)
        res3 = retval3.to_numpy_array()
        print(res3)
        assert retval3.shape == b.shape


time_end = time.time()

print("Seconds: "+str(time_end-time_start))

print("*"*80)
print("FIN")
print("*"*80)

backend.set_backend('scipy')

import importlib
importlib.reload(libpdefd_vector_array)
importlib.reload(libpdefd_matrix_compute)



"""
Problem size
"""
N = 2

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
a = libpdefd_vector_array.vector_array_zeros((N,N))
al = np.prod(a.shape)



for z in range(a.shape[0]):
    for y in range(a.shape[1]):
        a[z,y] = y*a.shape[1] + z*(a.shape[1]*a.shape[0])


print("")
print("*"*80)
print("Array B")
print("*"*80)
b = libpdefd_vector_array.vector_array_zeros((N,N))
bl = np.prod(b.shape)

for z in range(b.shape[0]):
    for y in range(b.shape[1]):
        b[z,y] = 5 + y*b.shape[1] + z*(b.shape[1]*b.shape[0])


print("")
print("*"*80)
print("Array C")
print("*"*80)
c = b*2.0
c = c.flatten()
print(c)



print("")
print("*"*80)
print("Matrix Sparse")
print("*"*80)

print(" + allocation")
m_setup = libpdefd_matrix_setup.matrix_sparse(shape=(bl, al))

print(" + setup")
for i in range(min(al, bl)):
    m_setup[i,i] = -2

for i in range(min(al, bl)-1):
    m_setup[i,i+1] = 1
    m_setup[i,i-1] = 3

print(m_setup)


print("*"*80)
print("Matrix Compute")
print("*"*80)

m_compute = libpdefd_matrix_compute.matrix_sparse(m_setup)
print(m_compute.to_numpy_array())
print(a.to_numpy_array())

print("*"*80)
print("Benchmarks")
print("*"*80)

time_start = time.time()

for k in range(K):
    print("Iteration: "+str(k))

    if 1:
        print("MUL test 1")
        retval = m_compute.dot__DEPRECATED(a.flatten())
        print(retval.to_numpy_array())
        assert retval.shape == c.shape
        assert np.isclose(retval.to_numpy_array() - res1, 0).all()

    if 1:
        print("MUL test 2")
        retval = m_compute.dot_add_reshape(a, c, b.shape)
        print(retval.to_numpy_array())
        assert retval.shape == b.shape
        assert np.isclose(retval.to_numpy_array() - res2, 0).all()



    if 1:
        print("MUL test 3")
        retval = m_compute.dot_add_reshape(a, c, b.shape)
        print(retval.to_numpy_array())
        assert retval.shape == b.shape
        assert np.isclose(retval.to_numpy_array() - res3, 0).all()




time_end = time.time()

print("Seconds: "+str(time_end-time_start))

print("*"*80)
print("FIN")
print("*"*80)
