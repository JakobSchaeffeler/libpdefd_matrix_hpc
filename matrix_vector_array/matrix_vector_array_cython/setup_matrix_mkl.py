from setuptools import setup, Extension
from Cython.Build import cythonize
import os
MKLROOT = os.environ['MKLROOT']
CONDA_PREFIX = os.environ['CONDA_PREFIX']
np_path = CONDA_PREFIX + '/lib/python3.8/site-packages/numpy/core/include'
print(np_path)
sched_path = '/usr/include'
mkl_include = MKLROOT + '/include'
ipp_path= '/shared/intel/2020/compilers_and_libraries_2020.2.254/linux/ipp/include/'
#ipp_path = '/usr/include'
print(mkl_include)


ext_modules = [
    Extension(
        "libpdefd_matrix_compute_cython_mkl",
        ["libpdefd_matrix_compute_cython_mkl.pyx"],
        extra_compile_args=['-fopenmp', '-lirc'],
        extra_link_args=['-fopenmp', '-lirc'],
        include_dirs=['.', '../', 'mkl_naiv', 'mkl_typed', 'mkl_malloc', 'mkl_cpp_alloc', np_path, mkl_include ]
    )
]


setup(
        name='libpdefd_matrix_compute_cython_mkl',
        ext_modules=cythonize(ext_modules, annotate=True, include_path=[".", "../", 'mkl_naiv', 'mkl_typed', 'mkl_malloc', 'mkl_cpp_alloc', np_path, mkl_include ])  # Python code file with primes() function

)
