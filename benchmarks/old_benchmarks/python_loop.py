import numpy as np
import sys
import os


def add(x,y,k):
    for _ in range(k):
        x + y
    return


def iadd(x,y,k):
    for _ in range(k):
        x.__iadd__(y)
    return


def radd(x,y,k):
    for _ in range(k):
        y + x
    return


def sub(x,y,k):
    for _ in range(k):
        x.__sub__(y)
    return


def isub(x,y,k):
    for _ in range(k):
        x.__isub__(y)
    return


def rsub(x,y,k):
    for _ in range(k):
        x.__rsub__(y)
    return


def mul(x,y,k):
    for _ in range(k):
        x.__mul__(y)
    return


def imul(x,y,k):
    for _ in range(k):
        x.__imul__(y)
    return


def rmul(x,y,k):
    for _ in range(k):
        x.__rmul__(y)
    return


def truediv(x,y,k):
    for _ in range(k):
        x.__truediv__(y)
    return


def itruediv(x,y,k):
    for _ in range(k):
        x.__itruediv__(y)
    return


def rtruediv(x,y,k):
    for _ in range(k):
        x.__rtruediv__(y)
    return


def kron_vector(x,y,k):
    for _ in range(k):
        x.kron_vector(y)
    return


def pow(x,y,k):
    for _ in range(k):
        x.__pow__(y)
    return


def add_matrix(x, y, k):
    for _ in range(k):
        x.__iadd__(y)
    return


def dot_add_reshape(A, x, y, dst_shape, k):
    for _ in range(k):
        A.dot_add_reshape(x,y,dst_shape)
