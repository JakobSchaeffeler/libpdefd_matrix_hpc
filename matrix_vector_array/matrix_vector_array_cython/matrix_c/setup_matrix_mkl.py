from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "libpdefd_matrix_compute_cython_mkl",
        ["libpdefd_matrix_compute_cython_mkl.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=['.', '../', '../mkl_naiv', '../mkl_typed', '../mkl_malloc', '../mkl_cpp_alloc', ]
    )
]


setup(
        name='libpdefd_matrix_compute_cython_mkl',
        ext_modules=cythonize(ext_modules, annotate=True, include_path=[".", "../", '../mkl_naiv', '../mkl_typed', '../mkl_malloc', '../mkl_cpp_alloc',  ])  # Python code file with primes() function

)
