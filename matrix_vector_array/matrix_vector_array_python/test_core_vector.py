#! /usr/bin/env python3

import sys, os
import time
import numpy as np
from ctypes import *
a, trash = os.path.split(os.path.abspath(__file__))
a, trash = os.path.split(a)
sys.path.append(a)
import backend
import importlib
backend.set_backend('mkl')
import libpdefd_vector_array as libpdefd_vector_array
import libpdefd_matrix_setup as libpdefd_matrix_setup
import libpdefd_matrix_compute as libpdefd_matrix_compute
sys.path.pop()

mkl = cdll.LoadLibrary("libmkl_rt.so")

mkl.mkl_set_num_threads(byref(c_int(1)))


funcs = ("__add__",
        "__iadd__",
        "__radd__",
        "__sub__",
        "__isub__",
        "__rsub__",
        "__mul__",
        "__imul__",
        "__rmul__",
        "__pow__",
        "__truediv__",
        "__rtruediv__",
        "__itruediv__",
        "kron_vector"
        )



"""
Problem size
"""
N = 80000

if len(sys.argv) > 1:
    N = int(sys.argv[1])

"""
Number of iterations
"""
K = 1

x = np.random.rand(N)
y = np.random.rand(N)

if len(sys.argv) > 2:
    K = int(sys.argv[2])

for func in funcs:
    print("Testing ", func)
    backend.set_backend('scipy')

    importlib.reload(libpdefd_vector_array)
    importlib.reload(libpdefd_matrix_compute)
    #np.random.seed(1234)

    scalar = np.random.rand()

    if func == "kron_vector":
        x = np.random.rand(int(N/100))
        y = np.random.rand(int(N/100))
    else:
        x = np.random.rand(N)
        y = np.random.rand(N)

    print("")
    print("*"*80)
    print("Array A")
    print("*"*80)
    a_np = libpdefd_vector_array.vector_array(x)

    print("")
    print("*"*80)
    print("Array B")
    print("*"*80)
    b_np = libpdefd_vector_array.vector_array(y)


    if(func == "kron_vector"):
        a_np = a_np.flatten()
        b_np = b_np.flatten()

    print("*"*80)
    print("Tests")
    print("*"*80)

    time_start = time.time()
    res1 = None
    res2 = None
    for k in range(K):
        print("Iteration: "+str(k))

        if 1:
            print("test 1")
            retval1 = eval("a_np." + func + "(b_np)")
            print(mkl.vmlGetErrStatus())

            res1 = retval1.to_numpy_array().copy()

        if func not in ("kron_vector"):
            print("test 2")
            retval2 = eval("a_np." + func + "(scalar)")
            print(mkl.vmlGetErrStatus())

            res2 = retval2.to_numpy_array().copy()


    time_end = time.time()

    print("Seconds: "+str(time_end-time_start))

    print("*"*80)
    print("FIN")
    print("*"*80)

    backend.set_backend('mkl')

    import importlib
    importlib.reload(libpdefd_vector_array)
    importlib.reload(libpdefd_matrix_compute)

    print("")
    print("*"*80)
    print("Array A")
    print("*"*80)
    a = libpdefd_vector_array.vector_array(x)


    print("")
    print("*"*80)
    print("Array B")
    print("*"*80)
    b = libpdefd_vector_array.vector_array(y)

    if(func == "kron_vector"):
        a = a.flatten()
        b = b.flatten()



    print("*"*80)
    print("Tests")
    print("*"*80)

    #mkl.vmlSetMode(c_uint(2561))

    time_start = time.time()
    for k in range(K):
        print("Iteration: "+str(k))

        if 1:
            print("test 1")
            retval1 = eval("a." + func + "(b)")


            assert np.isclose(retval1.to_numpy_array() - res1, 0).all()

        if func not in ("kron_vector"):
            print("test 2")
            retval2 = eval("a." + func + "(scalar)")
            print(retval2.to_numpy_array())
            print(res2)
            print(np.isclose(retval2.to_numpy_array() - res2, 0))

            # for i in range(N):
            #     if a._data_as[i] != a_np._data[i]:
            #         print(a.to_numpy_array()[i])
            #         print(a_np.to_numpy_array()[i])
            #         print("value at ", i, "different")
            #         raise Exception("diff val")

            if not np.isclose(retval2.to_numpy_array() - res2, 0, atol=1e-5, rtol=1e-3).all():
                for i in range(len(np.isclose(retval2.to_numpy_array() - res2, 0, atol=1e-5))):
                    if np.isclose(retval2.to_numpy_array() - res2, 0)[i] == False:
                        print(i)
                        print("scipy val: ", '{0:.64f}'.format(retval2.to_numpy_array()[i]))
                        print("mkl_val  : ", '{0:.64f}'.format(res2[i]))
                        print('{0:.64f}'.format(a.to_numpy_array()[i]))
                        print('{0:.64f}'.format(a_np.to_numpy_array()[i]))
                        print('{0:.64f}'.format(scalar))

            print(retval2.shape)
            assert np.isclose(retval2.to_numpy_array() - res2, 0, atol=1e-6).all()



    time_end = time.time()

    print("Seconds: "+str(time_end-time_start))

    print("*"*80)
    print("FIN")
    print("*"*80)

    backend.set_backend('mkl_blas')

    import importlib
    importlib.reload(libpdefd_vector_array)
    importlib.reload(libpdefd_matrix_compute)

    print("")
    print("*"*80)
    print("Array A")
    print("*"*80)
    a = libpdefd_vector_array.vector_array(x)

    print("")
    print("*"*80)
    print("Array B")
    print("*"*80)
    b = libpdefd_vector_array.vector_array(y)

    if(func == "kron_vector"):
        a = a.flatten()
        b = b.flatten()

    print("*"*80)
    print("Tests")
    print("*"*80)

    time_start = time.time()
    for k in range(K):
        print("Iteration: "+str(k))

        if 1:
            print("test 1")
            retval1 = eval("a." + func + "(b)")
            assert np.isclose(retval1.to_numpy_array() - res1, 0).all()

        if func not in ("kron_vector"):
            print("test 2")
            retval2 = eval("a." + func + "(scalar)")
            if not np.isclose(retval2.to_numpy_array() - res2, 0).all():
                for i in range(len(np.isclose(retval2.to_numpy_array() - res2, 0))):
                    if np.isclose(retval2.to_numpy_array() - res2, 0)[i] == False:
                        print(i)
                        print("scipy val: ",'{0:.64f}'.format(retval2.to_numpy_array()[i]))
                        print("mkl_val  : ",'{0:.64f}'.format(res2[i]))
                        print('{0:.64f}'.format(a.to_numpy_array()[i]))
                        print('{0:.64f}'.format(a_np.to_numpy_array()[i]))
                        print('{0:.64f}'.format(scalar))

            assert np.isclose(retval2.to_numpy_array() - res2, 0).all()




    time_end = time.time()

    print("Seconds: "+str(time_end-time_start))

    print("*"*80)
    print("FIN")
    print("*"*80)
