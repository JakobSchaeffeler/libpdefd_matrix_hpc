from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Compiler import Options

Options.cimport_from_pyx = True

ext_modules = [
    Extension(
        "cython_loop",
        ["cython_loop.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'], include_dirs=["../matrix_vector_array/matrix_vector_array_cython/mkl_c", "../matrix_vector_array/matrix_vector_array_cython", "../matrix_vector_array/matrix_vector_array_cython/mkl_blas_c", "../matrix_vector_array"]
    )
]


setup(
        name='cython_loop',
        ext_modules=cythonize(ext_modules, annotate=True, include_path=["../matrix_vector_array/matrix_vector_array_cython/mkl_c", ".", "../matrix_vector_array/matrix_vector_array_cython", "../matrix_vector_array/matrix_vector_array_cython/mkl_blas_c", "../matrix_vector_array"]),

        # Python code file with primes() function
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
