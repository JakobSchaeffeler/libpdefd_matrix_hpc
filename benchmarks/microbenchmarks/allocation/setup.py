from setuptools import setup
from Cython.Build import cythonize

setup(
    name='allocations',
    ext_modules=cythonize("allocations.pyx"),
    zip_safe=False,
)
