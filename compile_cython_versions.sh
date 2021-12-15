cd matrix_vector_array
rm *cpython*
cd matrix_vector_array_cython
rm *cpython*

cd mkl_cpp_alloc
rm -r build
rm *cpython*
LDSHARED="icpc -shared" CC=icpc python setup_vector_mkl_intel.py build_ext --inplace
cp libpdefd_vector_array_cython_mkl_cpp_alloc.cpython* ../
cp libpdefd_vector_array_cython_mkl_cpp_alloc.cpython* ../../
cd ../

cd mkl_malloc
rm -r build
rm *cpython*

#hack to make cython compilation successful
python setup_vector_mkl.py build_ext --inplace
rm -r build
rm *cpython*

LDSHARED="icpc -shared" CC=icpc python setup_vector_mkl_intel.py build_ext --inplace
cp libpdefd_vector_array_cython_mkl_malloc.cpython* ../
cp libpdefd_vector_array_cython_mkl_malloc.cpython* ../../
cd ../

cd mkl_naiv
rm -r build
rm *cpython*
LDSHARED="icpc -shared" CC=icpc python setup_vector_mkl_intel.py build_ext --inplace
cp libpdefd_vector_array_cython_mkl.cpython* ../
cp libpdefd_vector_array_cython_mkl.cpython* ../../
cd ../

cd mkl_typed
rm -r build
rm *cpython*

#hack to make cython compilation successful
python setup_vector_mkl.py build_ext --inplace
rm -r build
rm *cpython*

LDSHARED="icpc -shared" CC=icpc python setup_vector_mkl_intel.py build_ext --inplace
cp libpdefd_vector_array_cython_mkl_typed.cpython* ../
cp libpdefd_vector_array_cython_mkl_typed.cpython* ../../
cd ../

LDSHARED="icpc -shared" CC=icpc python setup_matrix_mkl.py build_ext --inplace



