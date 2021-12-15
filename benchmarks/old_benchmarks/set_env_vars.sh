export KMP_BLOCKTIME=30
#export OMP_PLACES=cores
#export OMP_NUM_THREADS=24

#export GOMP_CPU_AFFINITY="0,1,2,6,7,8,3,4,5,9,10,11,12,13,14,18,19,20,15,16,17,21,22,23"
#export GOMP_CPU_AFFINITY="0-23"
#export OMP_SCHEDULE=STATIC
#export OMP_PROC_BIND=CLOSE

#export KMP_HW_SUBSET=1t
export OMP_DEBUG=enabled
#export KMP_AFFINITY=compact,norespect,verbose,granularity=core,1,0
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
#export KMP_AFFINITY="granularity=fine,verbose,proclist=[0-23],explicit"
#export KMP_AFFINITY="verbose"
export MKL_VERBOSE=0
#export KMP_TOPOLOGY_METHOD="hwloc"

#export KMP_AFFINITY=proclist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],explicit

export KMP_SETTINGS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_NESTED=TRUE
export MKL_NESTED=TRUE
#export OMP_PROC_BIND=TRUE
#export OMP_NUM_TEAMS=4
#export KMP_STACKSIZE=400M
#export KMP_SCHEDULE=static,balanced
