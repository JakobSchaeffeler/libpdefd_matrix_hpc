import allocations
import tracemalloc


# ... run your application ...

#print(allocations.run_mkl_malloc()/1000000)
#print("finished")
#wait(10)
print(allocations.run_np_malloc()/1000000)


