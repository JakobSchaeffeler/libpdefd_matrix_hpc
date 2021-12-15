from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
CONDA_PREFIX = os.environ['CONDA_PREFIX']

np_path = CONDA_PREFIX + '/lib/python3.8/site-packages/numpy/core/include'

MKLROOT = os.environ['MKLROOT']
mkl_include = MKLROOT + '/include'


ext_modules = [
    Extension(
        "allocations",
        ["allocations.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs =[np_path, mkl_include]
 
        )
]


setup(
        name='allocations',
        ext_modules=cythonize(ext_modules, annotate=True,  # Python code file with primes() function
        compiler_directives={
            'embedsignature':False,
            'language_level':3,
            'c_string_type':'str',
            'c_string_encoding':'ascii',
            'py2_import':False,
            'nonecheck':False,
            'boundscheck':False,
            'wraparound':False},
            include_path=[mkl_include, np_path,]
        
        
        )

)
