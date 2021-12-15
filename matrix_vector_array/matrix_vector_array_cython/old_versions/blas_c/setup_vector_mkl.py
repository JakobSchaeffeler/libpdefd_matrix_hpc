from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "libpdefd_vector_array_cython_blas_c",
        ["libpdefd_vector_array_cython_blas_c.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]


setup(
        name='libpdefd_vector_array_cython_blas_c',
        ext_modules=cythonize(ext_modules, annotate=True)  # Python code file with primes() function
#        annotate=True),

#        ext_modules = cythonize("libpdefd_vector_array_mkl.pyx")
#  name = "libpdefd_vector_array_mkl",
# cmdclass = {"build_ext": build_ext},
#  ext_modules =
#  [Extension("libpdefd_vector_array_mkl",
#              ["libpdefd_vector_array_mkl.pyx"],
#              extra_compile_args = ["-O0", "-fopenmp"],
#              extra_link_args=['-fopenmp']
#              )]

)
