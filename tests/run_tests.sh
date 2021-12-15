
for op in "reduce_min" "reduce_minabs" "pos" "neg" "abs" "iadd" "add" "radd" "mul" "imul" "rmul" "sub" "isub" "rsub" "pow" "truediv" "itruediv" "rtruediv"
do
	#for N in "2" "64" "1024" "8192" "65536"  
	for N in "80000" 
	do
		#for version in "scipy" 
		#for version in "mkl_blas_cython_cpp" 
		#for version in "scipy" "mkl" "mkl_cython_typed" "mkl_cython_cpp"
		for version in "scipy" "mkl" "mkl_cython_cpp" 
		do
			for cores in "24"
			do
				export MKL_NUM_THREADS=$cores
				export OMP_NUM_THREADS=$cores
				export PYSM_BACKEND=$version 
				./tests_vector.py $N $version $op
			done
		done
	done
done
