#export KMP_BLOCKTIME=30
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

#export KMP_AFFINITY=norespect,proclist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],explicit
#export MKL_VERBOSE=0
export KMP_SETTINGS=1
#export OMP_DYNAMIC=FALSE
#export MKL_DYNAMIC=FALSE
