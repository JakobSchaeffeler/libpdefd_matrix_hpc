#export KMP_BLOCKTIME=30
export OMP_PLACES=cores
export GOMP_CPU_AFFINITY="0-23"
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE

#export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
#export KMP_SETTINGS=1
#export MKL_VERBOSE=1  
