#! /usr/bin/env python3

import sys, os
import time
import numpy as np
import importlib
from ctypes import *
import mkl_cpp_alloc.libpdefd_vector_array_cython_mkl_cpp_alloc as libpdefd_vector_array_cython_mkl_typed_cdef



import np.libpdefd_vector_array_cython_np as libpdefd_vector_array_cython_np
mkl = cdll.LoadLibrary("libmkl_rt.so")


os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'




"""
Problem size
"""
N = 80000



a = np.random.rand(N,)
a_np = libpdefd_vector_array_cython_np.vector_array(a)
a_mkl = libpdefd_vector_array_cython_mkl.vector_array(a)
a_blas = libpdefd_vector_array_cython_blas.vector_array(a)
a_mkl_c = libpdefd_vector_array_cython_mkl_c.vector_array(a)
a_blas_c = libpdefd_vector_array_cython_blas_c.vector_array(a)
a_mkl_blas_c = libpdefd_vector_array_cython_mkl_blas_c.vector_array(a)
a_mkl_malloc = libpdefd_vector_array_cython_mkl_malloc.vector_array(a)
a_typed = libpdefd_vector_array_cython_mkl_typed.vector_array(a)
a_typed_cdef = libpdefd_vector_array_cython_mkl_typed_cdef.vector_array(a)

b = np.random.rand(N,)
b_np = libpdefd_vector_array_cython_np.vector_array(b)
b_mkl = libpdefd_vector_array_cython_mkl.vector_array(b)
b_blas = libpdefd_vector_array_cython_blas.vector_array(b)
b_mkl_c = libpdefd_vector_array_cython_mkl_c.vector_array(b)
b_blas_c = libpdefd_vector_array_cython_blas_c.vector_array(b)
b_mkl_blas_c = libpdefd_vector_array_cython_mkl_blas_c.vector_array(b)
b_mkl_malloc = libpdefd_vector_array_cython_mkl_malloc.vector_array(b)
b_typed = libpdefd_vector_array_cython_mkl_typed.vector_array(b)
b_typed_cdef = libpdefd_vector_array_cython_mkl_typed_cdef.vector_array(b)


"""
Testing add
"""

c_np = a_np + b_np
c_mkl = a_mkl + b_mkl
c_blas = a_blas + b_blas
c_mkl_c = a_mkl_c + b_mkl_c
c_blas_c = a_blas_c + b_blas_c
c_mkl_blas_c = a_mkl_blas_c + b_mkl_blas_c
c_mkl_malloc = a_mkl_malloc + b_mkl_malloc
c_typed = a_typed + b_typed
c_typed_cdef = a_typed_cdef + b_typed_cdef
assert np.isclose(c_np.to_numpy_array() - c_mkl.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_blas.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_malloc.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_typed.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_typed_cdef.to_numpy_array(), 0).all()

1.51 + a_mkl_blas_c


c_np = a_np + 1.51
c_mkl = a_mkl + 1.51
c_blas = a_blas + 1.51
c_mkl_c = a_mkl_c + 1.51
c_blas_c = a_blas_c + 1.51
c_mkl_blas_c = a_mkl_blas_c + 1.51
c_mkl_malloc = a_mkl_malloc + 1.51
c_typed = a_typed + 1.51
c_typed_cdef = a_typed_cdef + 1.51

assert np.isclose(c_np.to_numpy_array() - c_mkl.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_blas.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_malloc.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_typed.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_typed_cdef.to_numpy_array(), 0).all()

"""
Testing iadd
"""


a_np += b_np
a_mkl += b_mkl
a_blas += b_blas
a_mkl_c += b_mkl_c
a_blas_c += b_blas_c
a_mkl_blas_c += b_mkl_blas_c
a_mkl_malloc += b_mkl_malloc
a_typed += b_typed
a_typed_cdef += b_typed_cdef



assert np.isclose(a_np.to_numpy_array() - a_mkl.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_blas.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_mkl_c.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_blas_c.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_mkl_malloc.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_typed.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_typed_cdef.to_numpy_array(), 0).all()

a_np += 1.51
a_mkl += 1.51
a_blas += 1.51
a_mkl_c += 1.51
a_mkl_blas_c += 1.51
a_blas_c += 1.51
a_mkl_malloc += 1.51
a_typed += 1.51
a_typed_cdef += 1.51

assert np.isclose(a_np.to_numpy_array() - a_mkl.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_blas.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_mkl_c.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_blas_c.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_mkl_malloc.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_typed.to_numpy_array(), 0).all()
assert np.isclose(a_np.to_numpy_array() - a_typed_cdef.to_numpy_array(), 0).all()


"""
Testing mul
"""

c_np = a_np * b_np
c_mkl = a_mkl * b_mkl
#c_blas = a_blas * b_blas
c_mkl_c = a_mkl_c * b_mkl_c
#c_blas_c = a_blas_c * b_blas_c
c_mkl_blas_c = a_mkl_blas_c * b_mkl_blas_c
#c_mkl_malloc = a_mkl_malloc * b_mkl_malloc
#c_typed = a_typed * b_typed
#c_typed_cdef = a_typed_cdef * b_typed_cdef
#assert np.isclose(c_np.to_numpy_array() - c_mkl.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_blas_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_mkl_malloc.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed_cdef.to_numpy_array(), 0).all()



c_np = a_np * 1.51
#c_mkl = a_mkl * 1.51
#c_blas = a_blas * 1.51
c_mkl_c = a_mkl_c * 1.51
#c_blas_c = a_blas_c * 1.51
c_mkl_blas_c = a_mkl_blas_c * 1.51
#c_mkl_malloc = a_mkl_malloc * 1.51
#c_typed = a_typed * 1.51
#c_typed_cdef = a_typed_cdef * 1.51

#assert np.isclose(c_np.to_numpy_array() - c_mkl.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_blas_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_mkl_malloc.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed_cdef.to_numpy_array(), 0).all()



"""
Testing imul
"""

a_np *= b_np
#a_mkl *= b_mkl
#a_blas *= b_blas
a_mkl_c *= b_mkl_c
#a_blas_c *= b_blas_c
a_mkl_blas_c *= b_mkl_blas_c
#a_mkl_malloc *= b_mkl_malloc
#a_typed *= b_typed
#a_typed_cdef *= b_typed_cdef
#assert np.isclose(c_np.to_numpy_array() - c_mkl.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_blas_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_mkl_malloc.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed_cdef.to_numpy_array(), 0).all()



a_np *= 1.51
#a_mkl *= 1.51
#a_blas *= 1.51
a_mkl_c *= 1.51
#a_blas_c *= 1.51
a_mkl_blas_c *= 1.51
#a_mkl_malloc *= 1.51
#a_typed *= 1.51
#a_typed_cdef *= 1.51

#assert np.isclose(c_np.to_numpy_array() - c_mkl.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_blas_c.to_numpy_array(), 0).all()
assert np.isclose(c_np.to_numpy_array() - c_mkl_blas_c.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_mkl_malloc.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed.to_numpy_array(), 0).all()
#assert np.isclose(c_np.to_numpy_array() - c_typed_cdef.to_numpy_array(), 0).all()






reps = 100000
if 0:
#for i in [1,4]:
    print(i, " cores")
    mkl.mkl_set_num_threads(byref(c_int(i)))


    start = time.time()
    for _ in range(reps):
        c_np = a_np + b_np
    end = time.time()

    print("np:", end-start)

    # start = time.time()
    # for _ in range(reps):
    #     c_blas = a_blas + b_blas
    # end = time.time()
    # print("blas:", end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     c_blas_c = a_blas_c + b_blas_c
    # end = time.time()
    #
    # print("blas_c:",end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     c_mkl_blas_c = a_mkl_blas_c + b_mkl_blas_c
    # end = time.time()
    #
    # print("mkl_blas_c:", end-start)
    #
    #
    start = time.time()
    for _ in range(reps):
        c_mkl = a_mkl + b_mkl
    end = time.time()

    print("mkl:", end-start)

    start = time.time()
    for _ in range(reps):
        c_typed = a_typed + b_typed
    end = time.time()

    print("mkl_typed:", end-start)


    start = time.time()
    for _ in range(reps):
        c_typed_cdef = a_typed_cdef + b_typed_cdef
    end = time.time()

    print("mkl_typed_cdef:", end-start)

    # start = time.time()
    # for _ in range(reps):
    #     c_mkl_c = a_mkl_c + b_mkl_c
    # end = time.time()
    #
    # print("mkl_c:", end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     c_mkl_malloc = a_mkl_malloc + b_mkl_malloc
    # end = time.time()
    #
    # print("mkl_c_malloc:", end-start)

    print("iadd:")

    start = time.time()
    for _ in range(reps):
        a_np += b_np
    end = time.time()

    print("np :", end-start)

    start = time.time()
    for _ in range(reps):
        a_mkl += b_mkl
    end = time.time()

    print("mkl: ", end-start)

    start = time.time()
    for _ in range(reps):
        a_typed += b_typed
    end = time.time()

    print("mkl_typed:", end-start)


    start = time.time()
    for _ in range(reps):
        a_typed_cdef += b_typed_cdef
    end = time.time()

    print("mkl_typed_cdef:", end-start)


    # start = time.time()
    # for _ in range(reps):
    #     a_blas += b_blas
    # end = time.time()
    # print(end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     a_blas_c += b_blas_c
    # end = time.time()
    # print(end-start)
    #
    #
    #
    # print(end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     a_mkl_c += b_mkl_c
    # end = time.time()
    #
    # print(end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     a_mkl_malloc += b_mkl_malloc
    # end = time.time()
    #
    # print(end-start)

    print("add:")

    start = time.time()
    for _ in range(reps):
        c_np = a_np + 1.51
    end = time.time()

    print("np scalar:", end-start)

    # start = time.time()
    # for _ in range(reps):
    #     c_blas = a_blas + b_blas
    # end = time.time()
    # print("blas:", end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     c_blas_c = a_blas_c + b_blas_c
    # end = time.time()
    #
    # print("blas_c:",end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     c_mkl_blas_c = a_mkl_blas_c + b_mkl_blas_c
    # end = time.time()
    #
    # print("mkl_blas_c:", end-start)
    #
    #
    start = time.time()
    for _ in range(reps):
        c_mkl = a_mkl + 1.51
    end = time.time()

    print("mkl scalar:", end-start)

    start = time.time()
    for _ in range(reps):
        c_typed = a_typed + 1.51
    end = time.time()

    print("mkl_typed scalar:", end-start)


    start = time.time()
    for _ in range(reps):
        c_typed_cdef = a_typed_cdef + 1.51
    end = time.time()

    print("mkl_typed_cdef scalar:", end-start)

    # start = time.time()
    # for _ in range(reps):
    #     c_mkl_c = a_mkl_c + b_mkl_c
    # end = time.time()
    #
    # print("mkl_c:", end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     c_mkl_malloc = a_mkl_malloc + b_mkl_malloc
    # end = time.time()
    #
    # print("mkl_c_malloc:", end-start)

    print("iadd:")

    start = time.time()
    for _ in range(reps):
        a_np += 1.51
    end = time.time()

    print("np scalar:", end-start)

    start = time.time()
    for _ in range(reps):
        a_mkl += 1.51
    end = time.time()

    print("mkl scalar: ", end-start)

    start = time.time()
    for _ in range(reps):
        a_typed += 1.51
    end = time.time()

    print("mkl_typed scalar:", end-start)


    start = time.time()
    for _ in range(reps):
        a_typed_cdef += 1.51
    end = time.time()

    print("mkl_typed_cdef scalar:", end-start)


    # start = time.time()
    # for _ in range(reps):
    #     a_blas += b_blas
    # end = time.time()
    # print(end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     a_blas_c += b_blas_c
    # end = time.time()
    # print(end-start)
    #
    #
    #
    # print(end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     a_mkl_c += b_mkl_c
    # end = time.time()
    #
    # print(end-start)
    #
    # start = time.time()
    # for _ in range(reps):
    #     a_mkl_malloc += b_mkl_malloc
    # end = time.time()
    #
    # print(end-start)
