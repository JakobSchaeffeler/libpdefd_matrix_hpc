#setup script in order to be able to cimport modules not in the same dir from https://github.com/cython/cython/wiki/PackageHierarchy


# build script for 'dvedit' - Python libdv wrapper

# change this as needed
libdvIncludeDir = ""

import sys, os
from distutils.core import setup
from distutils.extension import Extension

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)

dir_cython = "../matrix_vector_array/matrix_vector_array_cython"


# scan the 'dvedit' directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir_cython, files=[]):
    for file in os.listdir(dir_cython):
        path = os.path.join(dir_cython, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [libdvIncludeDir, "."],   # adding the '.' to include_dirs is CRUCIAL!!
        extra_compile_args = ["-O3", "-Wall"],
        extra_link_args = ['-g'],
        libraries = ["dv",],
        )

# get the list of extensions
extNames = scandir(".")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

# finally, we can pass all this to distutils
setup(
  name="cython_loop",
  packages=["cython_loop", "matrix_vector_array_cython_mkl_c"],
  ext_modules=extensions,
  extra_compile_args=['-fopenmp'],
  extra_link_args=['-fopenmp'],
  cmdclass = {'build_ext': build_ext},
)

