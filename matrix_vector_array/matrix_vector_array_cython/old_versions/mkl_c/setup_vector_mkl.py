from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "libpdefd_vector_array_cython_mkl_c",
        ["libpdefd_vector_array_cython_mkl_c.pyx"],
        extra_compile_args=[ '-fopenmp', '-DMKL_DIRECT_CALL'],
    )
]


setup(
        name='libpdefd_vector_array_cython_mkl_c',
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
