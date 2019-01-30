import os,sys

DIR=sys.argv[1]
data = [os.path.join(DIR,f) for f in os.listdir(DIR) if f.startswith("ssnetout_larcv_") and f.endswith(".root")]

from multiprocessing.dummy import Pool as ThreadPool

def work(d):
    num = int(os.path.basename(d).split(".")[0].split("_")[-1])
    SS="python ../determine.py %s ana 1>stdout/out_%d.log 2>stderr/err_%d.log" %(d,num,num)
    print SS
    os.system(SS)
    return

# Make the Pool of workers
pool = ThreadPool(12)

results = pool.map(work, data)

pool.close()
pool.join()
