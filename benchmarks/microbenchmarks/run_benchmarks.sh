
for op in "init"
do
	for N in "80000" "128000000" 
	do
		for cores in {1..24}
		do
			export MKL_NUM_THREADS=$cores
			export OMP_NUM_THREADS=$cores
			./microbenchmarks.py $cores $N $op
		done
	done
done
