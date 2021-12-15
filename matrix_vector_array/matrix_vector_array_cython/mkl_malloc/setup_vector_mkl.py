from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "libpdefd_vector_array_cython_mkl_malloc",
        ["libpdefd_vector_array_cython_mkl_malloc.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
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
            'wraparound':False})

)
