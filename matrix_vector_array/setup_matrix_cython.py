from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "libpdefd_matrix_compute_cython",
        ["libpdefd_matrix_compute_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],  include_dirs=["matrix_vector_array_cython/mkl_c", "matrix_vector_array_cython", "matrix_vector_array_cython/mkl_blas_c"]
    )
]


setup(
    name='libpdefd_matrix_compute_cython',
    ext_modules=cythonize(ext_modules, annotate=True,
    include_path=["matrix_vector_array_cython/mkl_c", "matrix_vector_array_cython", "matrix_vector_array_cython/mkl_blas_c"]

        )
)
