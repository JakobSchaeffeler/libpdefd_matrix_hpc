
#for op in "iadd" "mul" "imul" "sub" "pow" "truediv" "itruediv"
#for op in "add" "iadd" "mul" "imul" "sub" "isub" "pow" "truediv" "itruediv" "rtruediv" "neg" "pos" "abs" 
#for op in "pow" "rtruediv" "truediv" "itruediv" "neg" "pos" "abs" 
for op in "iadd"   
do
	for N in "80000"
	do
		for version in "scipy" "mkl" "mkl_cython_cpp" 
		do
			for cores in {1..24}
			do
				export MKL_NUM_THREADS=$cores
				export OMP_NUM_THREADS=$cores
				export PYSM_BACKEND=$version 
				./benchmark_matrix.py $cores $N $version $op
			done
		done
	done
done
