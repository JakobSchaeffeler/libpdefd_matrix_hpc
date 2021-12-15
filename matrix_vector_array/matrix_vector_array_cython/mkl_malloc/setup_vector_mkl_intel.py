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
        "libpdefd_vector_array_cython_mkl_malloc",
        ["libpdefd_vector_array_cython_mkl_malloc.pyx",],
        extra_compile_args=['-Wl,--no-as-needed', '-lmkl_intel_ilp64', '-lmkl_intel_thread', '-I/usr/include', '-lmkl_core', '-liomp5', '-lpthread', '-lm', '-ldl',  '-DMKL_ILP64',  '-m64', '-qopenmp', '-lirc'],
        extra_link_args=['-I/usr/include', '-lmkl_intel_ilp64', '-lmkl_intel_thread', '-lmkl_core', '-liomp5', '-lpthread', '-lm', '-ldl',  '-DMKL_ILP64',  '-m64', '-fopenmp', '-lnuma',  '-lirc'],
    include_dirs =[sched_path, mkl_include, np_path, ipp_path,]
        )
]

setup(
        name='libpdefd_vector_array_cython_mkl_malloc',
        ext_modules=cythonize(ext_modules, annotate=True,  # Python code file with primes() function
        compiler_directives={
        'embedsignature':False,
        'language_level':3,
        'c_string_type':'str',
        'c_string_encoding':'ascii',
        'py2_import':False,
        'nonecheck':False,
        'boundscheck':False,
        'wraparound':False}, include_path=[sched_path, mkl_include, np_path,ipp_path,])

)

