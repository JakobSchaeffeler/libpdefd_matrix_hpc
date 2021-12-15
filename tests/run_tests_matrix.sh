
#for op in "iadd" "mul" "imul" "sub" "pow" "truediv" "itruediv"
#for op in "add" "iadd" "mul" "imul" "sub" "isub" "pow" "truediv" "itruediv" "rtruediv" "neg" "pos" "abs" 
#for op in "pow" "rtruediv" "truediv" "itruediv" "neg" "pos" "abs" 
for op in "iadd" "dot_add_reshape"
#for op in "add"
do
	#for N in "2" "64" "1024" "8192" "65536"  
	for N in "8000"
	do
		#for version in "scipy" 
		for version in "scipy" "mkl" "mkl_cython_cpp" 
		#for version in "mkl_cython_cpp"
		do
			for cores in "24"
			do
				export MKL_NUM_THREADS=$cores
				export OMP_NUM_THREADS=$cores
				export PYSM_BACKEND=$version 
				./tests_matrix.py $N $version $op
			done
		done
	done
done
