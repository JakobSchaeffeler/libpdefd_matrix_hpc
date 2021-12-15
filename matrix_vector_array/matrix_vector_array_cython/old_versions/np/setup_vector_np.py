from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "libpdefd_vector_array_cython_np",
        ["libpdefd_vector_array_cython_np.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]


setup(
        name='libpdefd_vector_array_cython_np',
        ext_modules=cythonize(ext_modules),
        compiler_directives={
            'embedsignature':False,
            'language_level':3,
            'c_string_type':'str',
            'c_string_encoding':'ascii',
            'py2_import':False,
            'warning_errors':False,
            'warn_unreachable':False,
            'nonecheck':False,
            'boundscheck':False,
            'wraparound':False}


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
